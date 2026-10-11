use std::collections::BTreeMap;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamic_expressions::OperatorSet;
use dynamic_expressions::strings::StringTreeOptions;
#[cfg(not(target_arch = "wasm32"))]
use dynamic_expressions::strings::julia_float;
use num_traits::Float;

use crate::dataset::Dataset;
use crate::hall_of_fame::{HallOfFame, pareto_volume};
use crate::options::Options;

pub type LogCallback<T> = Arc<dyn Fn(&LogPayload<T>) + Send + Sync>;

/// SymbolicRegression.jl's `SRLogger`: step zero and every `log_interval` steps,
/// plus the terminal state. Nonpositive intervals disable the callback.
#[derive(Clone)]
pub struct SrLogger<T> {
    pub callback: LogCallback<T>,
    pub log_interval: isize,
    log_step: Arc<AtomicUsize>,
}
impl<T> SrLogger<T> {
    pub fn new(callback: impl Fn(&LogPayload<T>) + Send + Sync + 'static) -> Self {
        Self::with_interval(callback, 100)
    }
    pub fn with_interval(callback: impl Fn(&LogPayload<T>) + Send + Sync + 'static, log_interval: isize) -> Self {
        Self {
            callback: Arc::new(callback),
            log_interval,
            log_step: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub(crate) fn should_log(&self) -> bool {
        let step = self.log_step.fetch_add(1, Ordering::Relaxed);
        self.log_interval > 0 && step.is_multiple_of(self.log_interval as usize)
    }
}
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LogPayload<T> {
    pub num_evals: f64,
    /// Remaining population cycles per output, for callers that monitor search state.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub cycles_remaining: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub data: LogOutputs<T>,
}
#[cfg_attr(feature = "serde", derive(serde::Serialize), serde(untagged))]
#[derive(Clone, Debug)]
pub enum LogOutputs<T> {
    Single(LogOutput<T>),
    Multiple(BTreeMap<String, LogOutput<T>>),
}
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LogOutput<T> {
    pub population: LogPopulation,
    pub summaries: LogSummaries<T>,
    pub equations: BTreeMap<String, LogEquation<T>>,
}
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LogPopulation {
    pub complexities: Vec<usize>,
}
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LogSummaries<T> {
    pub min_loss: T,
    pub pareto_volume: f64,
}
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LogEquation<T> {
    pub loss: T,
    pub equation: String,
}

pub(crate) fn log_output<T: Float + Display, Ops: OperatorSet, const D: usize>(
    hall: &HallOfFame<T, Ops, D>,
    complexities: impl Iterator<Item = usize>,
    dataset: &Dataset<T>,
    options: &Options<T, Ops, D>,
) -> LogOutput<T> {
    let front: Vec<_> = hall.front().collect();
    let losses: Vec<_> = front.iter().map(|m| m.loss).collect();
    let front_complexities: Vec<_> = front.iter().map(|m| m.complexity).collect();
    LogOutput {
        population: LogPopulation {
            complexities: complexities.collect(),
        },
        summaries: LogSummaries {
            min_loss: losses.last().copied().unwrap_or_else(T::infinity),
            pareto_volume: pareto_volume(&losses, &front_complexities, options.maxsize, options.loss_scale),
        },
        equations: front
            .iter()
            .map(|member| {
                (
                    format!("complexity={}", member.complexity),
                    LogEquation {
                        loss: member.loss,
                        equation: member.expr.string(
                            options,
                            StringTreeOptions {
                                variable_names: Some(&dataset.variable_names),
                                julia_constants: true,
                                ..Default::default()
                            },
                        ),
                    },
                )
            })
            .collect(),
    }
}

pub(crate) fn save_to_file<T: Float + Display, Ops: OperatorSet, const D: usize>(
    hall: &HallOfFame<T, Ops, D>,
    nout: usize,
    output: usize,
    dataset: &Dataset<T>,
    options: &Options<T, Ops, D>,
    run_id: &str,
) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::fmt::Write;
        let directory = std::path::Path::new(options.output_directory.as_deref().unwrap_or("outputs")).join(run_id);
        let filename = if nout > 1 {
            format!("hall_of_fame_output{}.csv", output + 1)
        } else {
            "hall_of_fame.csv".to_string()
        };
        std::fs::create_dir_all(&directory)
            .map_err(|e| format!("cannot create output directory {}: {e}", directory.display()))?;
        let mut csv = String::from("Complexity,Loss,Equation\n");
        for member in hall.front() {
            let equation = member.expr.string(
                options,
                StringTreeOptions {
                    variable_names: Some(&dataset.variable_names),
                    julia_constants: true,
                    ..Default::default()
                },
            );
            writeln!(
                csv,
                "{},{},\"{}\"",
                member.complexity,
                julia_float(member.loss.to_string().parse().expect("float loss")),
                equation.replace('"', "\"\"")
            )
            .unwrap();
        }
        let backup_filename = format!("{filename}.bak");
        for name in [&filename, &backup_filename] {
            let path = directory.join(name);
            write_atomically(&directory, name, &csv)
                .map_err(|e| format!("cannot write hall of fame {}: {e}", path.display()))?;
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (hall, nout, output, dataset, options, run_id);
    Ok(())
}

/// Writes `contents` to `directory/name` through a temporary file in the same directory, so that
/// a reader such as PySR, which may read these files while a search runs, sees either the previous
/// file or the whole new one.
#[cfg(not(target_arch = "wasm32"))]
fn write_atomically(directory: &std::path::Path, name: &str, contents: &str) -> std::io::Result<()> {
    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);
    let temp = directory.join(format!(
        ".{name}.tmp-{}-{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    let written = std::fs::write(&temp, contents).and_then(|()| std::fs::rename(&temp, directory.join(name)));
    if written.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    written
}

/// Generates `yyyymmdd_HHMMSS_` followed by six characters from `0-9a-zA-Z`.
pub fn generate_run_id() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    let timestamp = {
        let now = unsafe { libc::time(std::ptr::null_mut()) };
        let mut tm = std::mem::MaybeUninit::<libc::tm>::uninit();
        #[cfg(unix)]
        assert!(!unsafe { libc::localtime_r(&now, tm.as_mut_ptr()) }.is_null());
        #[cfg(windows)]
        assert_eq!(unsafe { libc::localtime_s(tm.as_mut_ptr(), &now) }, 0);
        let tm = unsafe { tm.assume_init() };
        format!(
            "{:04}{:02}{:02}_{:02}{:02}{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec
        )
    };
    #[cfg(target_arch = "wasm32")]
    let timestamp = {
        let seconds = web_time::SystemTime::now()
            .duration_since(web_time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let days = seconds / 86400 + 719468;
        let era = days / 146097;
        let doe = days % 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let year = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = year + u64::from(month <= 2);
        format!(
            "{year:04}{month:02}{day:02}_{:02}{:02}{:02}",
            seconds / 3600 % 24,
            seconds / 60 % 60,
            seconds % 60
        )
    };
    const ALPHABET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let suffix: String = (0..6)
        .map(|_| ALPHABET[fastrand::usize(..ALPHABET.len())] as char)
        .collect();
    format!("{timestamp}_{suffix}")
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::write_atomically;

    #[test]
    fn write_atomically_replaces_the_file_and_leaves_no_temporary_file() {
        let dir = std::env::temp_dir().join(format!("sr-atomic-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for contents in ["first\n", "second, longer\n"] {
            write_atomically(&dir, "hall_of_fame.csv", contents).unwrap();
            assert_eq!(std::fs::read_to_string(dir.join("hall_of_fame.csv")).unwrap(), contents);
        }
        assert!(write_atomically(&dir.join("missing"), "hall_of_fame.csv", "x").is_err());
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["hall_of_fame.csv"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
