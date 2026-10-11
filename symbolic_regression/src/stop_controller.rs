use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use num_traits::Float;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::options::Options;

#[derive(Clone, Debug)]
pub(crate) struct StopController {
    max_evals: u64,
    timeout_in_seconds: f64,
    start_time: Instant,
    cancelled: Arc<AtomicBool>,
}

impl StopController {
    pub(crate) fn from_options<T: Float, Ops, const D: usize>(options: &Options<T, Ops, D>) -> Self {
        Self {
            max_evals: options.max_evals,
            timeout_in_seconds: options.timeout_in_seconds,
            start_time: Instant::now(),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        if self.cancelled.load(Ordering::Acquire) {
            return true;
        }
        if self.timeout_in_seconds > 0.0 && self.start_time.elapsed().as_secs_f64() >= self.timeout_in_seconds {
            self.cancelled.store(true, Ordering::Release);
            return true;
        }
        false
    }

    pub(crate) fn should_stop(&self, total_evals: f64) -> bool {
        if self.is_cancelled() {
            return true;
        }
        if self.max_evals > 0 && total_evals >= self.max_evals as f64 {
            return true;
        }
        false
    }
}

/// A per-search request for a graceful stop after the next completed population cycle.
/// Clones share the request; independent handles never stop one another's searches.
#[derive(Clone, Debug, Default)]
pub struct ExternalStop {
    requested: Arc<AtomicBool>,
}
impl ExternalStop {
    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }
    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }
}

/// An input source whose callback returns currently available bytes without waiting.
/// Empty data, end-of-file, and `WouldBlock` mean that no command is available.
#[derive(Clone)]
pub enum InputStream {
    #[cfg(unix)]
    Stdin,
    Nonblocking(Arc<dyn Fn() -> std::io::Result<Vec<u8>> + Send + Sync>),
}
impl InputStream {
    pub fn nonblocking(read_available: impl Fn() -> std::io::Result<Vec<u8>> + Send + Sync + 'static) -> Self {
        Self::Nonblocking(Arc::new(read_available))
    }
    /// A channel-backed input stream for callers that receive input on another thread.
    pub fn channel() -> (Self, std::sync::mpsc::Sender<Vec<u8>>) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = std::sync::Mutex::new(receiver);
        let stream = Self::nonblocking(move || {
            let mut data = Vec::new();
            for chunk in receiver.lock().unwrap().try_iter() {
                data.extend(chunk);
            }
            Ok(data)
        });
        (stream, sender)
    }
    pub(crate) fn default_stdin() -> Option<Self> {
        #[cfg(unix)]
        {
            Some(Self::Stdin)
        }
        #[cfg(not(unix))]
        {
            None
        }
    }
}

pub(crate) struct StdinReader {
    input: Option<InputStream>,
    saw_quit_char: bool,
}
impl StdinReader {
    pub(crate) fn watch(input: Option<&InputStream>) -> Result<Self, String> {
        let input = match input {
            #[cfg(unix)]
            Some(InputStream::Stdin) => {
                use std::io::Read;
                use std::os::unix::fs::OpenOptionsExt;
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NONBLOCK)
                    .open("/dev/stdin")
                    .map_err(|e| format!("cannot monitor stdin: {e}"))?;
                let file = std::sync::Mutex::new(file);
                Some(InputStream::nonblocking(move || {
                    let mut bytes = Vec::new();
                    let mut buffer = [0; 4096];
                    let mut file = file.lock().unwrap();
                    loop {
                        match file.read(&mut buffer) {
                            Ok(0) => break,
                            Ok(n) => bytes.extend_from_slice(&buffer[..n]),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) => return Err(e),
                        }
                    }
                    Ok(bytes)
                }))
            }
            _ => input.cloned(),
        };
        let reader = Self {
            input,
            saw_quit_char: false,
        };
        reader.read_available().map_err(|e| e.to_string())?;
        Ok(reader)
    }
    fn read_available(&self) -> std::io::Result<Vec<u8>> {
        match &self.input {
            Some(InputStream::Nonblocking(callback)) => match callback() {
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::UnexpectedEof
                    ) =>
                {
                    Ok(Vec::new())
                }
                result => result,
            },
            #[cfg(unix)]
            Some(InputStream::Stdin) => unreachable!("stdin is opened when monitoring starts"),
            None => Ok(Vec::new()),
        }
    }
    pub(crate) fn check_for_quit(&mut self) -> Result<bool, String> {
        let data = self
            .read_available()
            .map_err(|e| format!("cannot read input stream: {e}"))?;
        for byte in data {
            match byte {
                3 => {
                    self.saw_quit_char = false;
                    return Ok(true);
                }
                b'q' => self.saw_quit_char = true,
                b'\r' | b'\n' if self.saw_quit_char => {
                    self.saw_quit_char = false;
                    return Ok(true);
                }
                b'\r' | b'\n' => {}
                _ => self.saw_quit_char = false,
            }
        }
        Ok(false)
    }
}
