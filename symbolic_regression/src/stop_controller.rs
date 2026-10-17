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
/// Empty data, end-of-file, and `WouldBlock` mean that no command is available; any other
/// error stops monitoring for the rest of the search.
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
    /// Starts monitoring `input`, discarding anything already waiting on it.
    pub(crate) fn watch(input: Option<&InputStream>) -> Self {
        let mut reader = Self {
            input: input.cloned(),
            saw_quit_char: false,
        };
        reader.read_available();
        reader
    }
    /// Returns the bytes available now. Like SymbolicRegression.jl, which ignores streams it
    /// cannot read, this stops monitoring once the input closes or fails.
    fn read_available(&mut self) -> Vec<u8> {
        let mut data = Vec::new();
        let open = match &self.input {
            #[cfg(unix)]
            Some(InputStream::Stdin) => read_stdin_available(&mut data),
            Some(InputStream::Nonblocking(callback)) => match callback() {
                Ok(bytes) => {
                    data = bytes;
                    true
                }
                Err(e) => matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::UnexpectedEof
                ),
            },
            None => true,
        };
        if !open {
            self.input = None;
        }
        data
    }
    pub(crate) fn check_for_quit(&mut self) -> bool {
        let data = self.read_available();
        for byte in data {
            match byte {
                3 => {
                    self.saw_quit_char = false;
                    return true;
                }
                b'q' => self.saw_quit_char = true,
                b'\r' | b'\n' if self.saw_quit_char => {
                    self.saw_quit_char = false;
                    return true;
                }
                b'\r' | b'\n' => {}
                _ => self.saw_quit_char = false,
            }
        }
        false
    }
}

/// Appends the bytes waiting on standard input to `bytes` without blocking, polling before each
/// read so that the descriptor's flags, which the parent shell shares, stay untouched. Returns
/// whether stdin is still readable: hang-up, end-of-file, and errors all end monitoring.
#[cfg(unix)]
fn read_stdin_available(bytes: &mut Vec<u8>) -> bool {
    let interrupted = || std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted;
    let mut buffer = [0u8; 4096];
    loop {
        let mut stdin = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `stdin` is one valid `pollfd`, and a zero timeout never waits.
        match unsafe { libc::poll(&mut stdin, 1, 0) } {
            0 => return true,
            ready if ready < 0 => {
                if interrupted() {
                    continue;
                }
                return false;
            }
            _ if stdin.revents & libc::POLLIN == 0 => return false,
            _ => {}
        }
        // SAFETY: `buffer` is writable for `buffer.len()` bytes.
        let read = unsafe { libc::read(libc::STDIN_FILENO, buffer.as_mut_ptr().cast(), buffer.len()) };
        match read {
            0 => return false,
            n if n > 0 => bytes.extend_from_slice(&buffer[..n as usize]),
            _ if interrupted() => {}
            _ => return false,
        }
    }
}
