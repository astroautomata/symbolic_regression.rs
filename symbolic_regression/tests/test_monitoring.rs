use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::array;
use symbolic_regression::prelude::*;
use symbolic_regression::{ExternalStop, InputStream, LogOutputs, LogPayload, SearchEngine, SrLogger, equation_search};

fn dataset() -> Dataset<f64> {
    let x = array![[1., 2., 3., 4., 5., 6., 7., 8.]];
    Dataset::new(x.clone(), x.row(0).mapv(|v| v * v))
}
fn options(niterations: usize) -> Options<f64, BuiltinOpsF64, 2> {
    Options {
        operators: BuiltinOpsF64::new().from_names(&["+", "*"]).unwrap(),
        populations: 1,
        population_size: 5,
        tournament_selection_n: 2,
        niterations,
        ncycles_per_iteration: 1,
        maxsize: 5,
        deterministic: true,
        seed: Some(0),
        should_optimize_constants: false,
        progress: false,
        verbosity: Some(0),
        save_to_file: false,
        input_stream: None,
        ..Default::default()
    }
}
type Captured = Arc<Mutex<Vec<LogPayload<f64>>>>;
fn logger(interval: isize) -> (SrLogger<f64>, Captured) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let records = captured.clone();
    (
        SrLogger::with_interval(move |payload| records.lock().unwrap().push(payload.clone()), interval),
        captured,
    )
}
#[test]
fn logger_receives_terminal_search_state() {
    let (log, records) = logger(1);
    let mut opts = options(1);
    opts.logger = Some(log);
    equation_search(&[dataset()], &opts).unwrap();
    assert_eq!(
        records
            .lock()
            .unwrap()
            .iter()
            .map(|p| p.cycles_remaining.clone())
            .collect::<Vec<_>>(),
        vec![vec![0]]
    );
    records.lock().unwrap().clear();
    equation_search(&[dataset(), dataset()], &opts).unwrap();
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].cycles_remaining.iter().sum::<usize>(), 1);
    assert_eq!(records[1].cycles_remaining, vec![0, 0]);
    assert_eq!(
        records
            .iter()
            .filter(|p| p.cycles_remaining.iter().all(|&r| r == 0))
            .count(),
        1
    );
    let LogOutputs::Multiple(outputs) = &records[1].data else {
        panic!("multiple outputs");
    };
    assert_eq!(
        outputs.keys().map(String::as_str).collect::<Vec<_>>(),
        ["output1", "output2"]
    );
}
#[test]
fn sr_logger_always_emits_normal_terminal_state() {
    for (iterations, interval, expected) in [(2, 100, 2), (3, 2, 2), (1, 0, 0), (5, -1, 0), (5, 2, 3)] {
        let (log, records) = logger(interval);
        let mut opts = options(iterations);
        opts.logger = Some(log);
        equation_search(&[dataset()], &opts).unwrap();
        let records = records.lock().unwrap();
        assert_eq!(records.len(), expected);
        if expected != 0 {
            assert_eq!(records.last().unwrap().cycles_remaining, vec![0]);
        }
    }
}
#[test]
fn logger_payload_matches_scalar_fields_and_evaluations() {
    let (log, records) = logger(1);
    let mut opts = options(1);
    opts.logger = Some(log);
    let result = equation_search(&[dataset()], &opts).unwrap();
    let records = records.lock().unwrap();
    let payload = &records[0];
    assert_eq!(payload.num_evals, result.total_evals);
    let LogOutputs::Single(output) = &payload.data else {
        panic!("single output");
    };
    assert_eq!(output.population.complexities.len(), 5);
    let front = result.halls_of_fame[0].pareto_front();
    assert_eq!(output.summaries.min_loss, front.last().unwrap().loss);
    assert_eq!(
        output.summaries.pareto_volume,
        symbolic_regression::pareto_volume(
            &front.iter().map(|m| m.loss).collect::<Vec<_>>(),
            &front.iter().map(|m| m.complexity).collect::<Vec<_>>(),
            opts.maxsize,
            opts.loss_scale
        )
    );
    for member in front {
        let equation = &output.equations[&format!("complexity={}", member.complexity)];
        assert_eq!(equation.loss, member.loss);
        assert_eq!(
            equation.equation,
            member.expr.string(
                &opts,
                dynamic_expressions::strings::StringTreeOptions {
                    julia_constants: true,
                    ..Default::default()
                }
            )
        );
    }
}
#[test]
fn external_stop_latches_direct_requests() {
    let stop = ExternalStop::default();
    assert!(!stop.is_requested());
    stop.request();
    assert!(stop.is_requested());
    assert!(stop.clone().is_requested());
    assert!(!ExternalStop::default().is_requested());
}
#[test]
fn external_stop_preserves_requests_before_search_entry() {
    let stop = ExternalStop::default();
    stop.request();
    let (log, records) = logger(1);
    let mut opts = options(20);
    opts.external_stop = Some(stop.clone());
    opts.logger = Some(log);
    let result = equation_search(&[dataset()], &opts).unwrap();
    assert_eq!(result.populations.len(), 1);
    assert_eq!(
        records
            .lock()
            .unwrap()
            .iter()
            .map(|p| p.cycles_remaining.clone())
            .collect::<Vec<_>>(),
        [vec![19]]
    );
    assert!(stop.is_requested());
}
#[test]
fn overlapping_searches_stop_independently() {
    let stop_a = ExternalStop::default();
    let stop_b = ExternalStop::default();
    let (log_a, records_a) = logger(1);
    let (log_b, records_b) = logger(1);
    let mut opts_a = options(1000);
    opts_a.external_stop = Some(stop_a.clone());
    opts_a.logger = Some(log_a);
    let mut opts_b = options(100);
    opts_b.external_stop = Some(stop_b.clone());
    opts_b.logger = Some(log_b);
    let ready = Arc::new(std::sync::Barrier::new(3));
    let resume = Arc::new(std::sync::Barrier::new(3));
    let searches: Vec<_> = [opts_a, opts_b]
        .into_iter()
        .map(|opts| {
            let ready = ready.clone();
            let resume = resume.clone();
            std::thread::spawn(move || {
                let mut engine = SearchEngine::new(vec![dataset()], opts).unwrap();
                engine.step(1).unwrap();
                assert!(!engine.is_finished());
                ready.wait();
                resume.wait();
                engine.run_to_completion().unwrap()
            })
        })
        .collect();
    ready.wait();
    stop_a.request();
    resume.wait();
    for search in searches {
        assert_eq!(search.join().unwrap().populations.len(), 1);
    }
    assert!(records_a.lock().unwrap().last().unwrap().cycles_remaining[0] > 0);
    assert_eq!(records_b.lock().unwrap().last().unwrap().cycles_remaining, [0]);
    assert!(stop_a.is_requested());
    assert!(!stop_b.is_requested());
}

#[test]
fn stdin_monitoring_is_nonblocking_and_parses_quit_commands() {
    for chunks in [
        vec!["q\r\n"],
        vec!["q", "\n"],
        vec!["\x03"],
        vec!["xq", "\n"],
        vec!["q", "x", "q\n"],
    ] {
        let (input, sender) = InputStream::channel();
        let mut opts = options(20);
        opts.input_stream = Some(input);
        let mut engine = SearchEngine::new(vec![dataset()], opts).unwrap();
        let started = Instant::now();
        engine.step(1).unwrap();
        assert!(started.elapsed() < Duration::from_secs(1));
        for (i, chunk) in chunks.iter().enumerate() {
            sender.send(chunk.as_bytes().to_vec()).unwrap();
            engine.step(1).unwrap();
            assert_eq!(engine.is_finished(), i + 1 == chunks.len());
        }
    }
    let (input, sender) = InputStream::channel();
    let mut opts = options(20);
    opts.input_stream = Some(input);
    let mut engine = SearchEngine::new(vec![dataset()], opts).unwrap();
    sender.send(b"q\n".to_vec()).unwrap();
    engine.step(1).unwrap();
    assert!(engine.is_finished());
}
#[test]
fn quit_monitor_ignores_non_quit_lines_and_stale_input() {
    let (input, sender) = InputStream::channel();
    sender.send(b"q\n".to_vec()).unwrap();
    let mut opts = options(20);
    opts.input_stream = Some(input);
    let mut engine = SearchEngine::new(vec![dataset()], opts).unwrap();
    engine.step(1).unwrap();
    assert!(!engine.is_finished());
    for chunk in [b"qz\n".as_slice(), b"normal\r\n", b"\n"] {
        sender.send(chunk.to_vec()).unwrap();
        engine.step(1).unwrap();
        assert!(!engine.is_finished());
    }
}
/// The default stdin monitor ignores a standard input it cannot read instead of failing the
/// search, as SymbolicRegression.jl does.
#[cfg(unix)]
#[test]
fn search_ignores_unreadable_stdin() {
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;
    use std::process::{Command, Stdio};

    const CHILD: &str = "SR_STDIN_MONITOR_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let mut opts = options(3);
        opts.input_stream = Options::<f64, BuiltinOpsF64, 2>::default().input_stream;
        assert!(opts.input_stream.is_some());
        equation_search(&[dataset()], &opts).unwrap();
        return;
    }
    let (open_socket, _peer) = UnixStream::pair().unwrap();
    let (hung_up_socket, closed_peer) = UnixStream::pair().unwrap();
    drop(closed_peer);
    for stdin in [open_socket, hung_up_socket] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "search_ignores_unreadable_stdin", "--test-threads=1"])
            .env(CHILD, "1")
            .stdin(Stdio::from(OwnedFd::from(stdin)))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
#[test]
fn stop_on_clock_time() {
    use rand::SeedableRng;
    use rand_distr::Distribution;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let x = ndarray::Array2::<f32>::from_shape_fn((5, 100), |_| rand_distr::StandardNormal.sample(&mut rng));
    let y = x.row(3).mapv(|v| 2.0 * v.cos());
    let data = Dataset::new(x, y);
    let mut opts = Options::<f32, BuiltinOpsF32, 2> {
        operators: BuiltinOpsF32::new().from_names(["cos", "+", "*"]).unwrap(),
        population_size: 10,
        tournament_selection_n: 9,
        ncycles_per_iteration: 100,
        maxsize: 15,
        timeout_in_seconds: 1.,
        populations: 1,
        niterations: 1,
        seed: Some(0),
        progress: false,
        verbosity: Some(0),
        save_to_file: false,
        input_stream: None,
        ..Default::default()
    };
    equation_search(std::slice::from_ref(&data), &opts).unwrap();
    opts.niterations = 10_000_000;
    let start = Instant::now();
    equation_search(std::slice::from_ref(&data), &opts).unwrap();
    assert!(start.elapsed().as_secs_f64() < 100.);
}

#[test]
fn max_evals_limits_current_search() {
    let mut opts = options(1000);
    opts.max_evals = 5;
    let result = equation_search(&[dataset()], &opts).unwrap();
    assert_eq!(result.total_evals, 5.);
}

#[test]
fn sr_logger_keeps_its_cadence_when_reused_across_searches() {
    let (log, records) = logger(100);
    let mut opts = options(1);
    opts.logger = Some(log);
    equation_search(&[dataset()], &opts).unwrap();
    assert_eq!(records.lock().unwrap().len(), 1);
    records.lock().unwrap().clear();
    opts.niterations = 2;
    equation_search(&[dataset()], &opts).unwrap();
    assert_eq!(records.lock().unwrap().len(), 1);
    assert_eq!(records.lock().unwrap()[0].cycles_remaining, [0]);
}

#[test]
fn sr_logger_defaults_to_an_interval_of_one_hundred() {
    let logger = SrLogger::new(|_: &LogPayload<f64>| {});
    assert_eq!(logger.log_interval, 100);
}

#[test]
fn logger_population_precedes_migration() {
    let (log, records) = logger(1);
    let mut opts = options(1);
    opts.logger = Some(log);
    opts.guesses = vec![(0..10).map(|_| Guess::from("x0 * x0 * x0 * x0")).collect()];
    opts.fraction_replaced_guesses = 10.0;
    let result = equation_search(&[dataset()], &opts).unwrap();
    assert!(
        result.populations[0][0]
            .members
            .iter()
            .any(|m| m.complexity > opts.maxsize),
        "the oversized guess migrates into the population after the cycle"
    );
    let records = records.lock().unwrap();
    let LogOutputs::Single(output) = &records[0].data else {
        panic!("single output");
    };
    assert!(
        output.population.complexities.iter().all(|&c| c <= opts.maxsize),
        "{:?}",
        output.population.complexities
    );
}

#[test]
fn logger_can_request_stop_after_its_completed_cycle() {
    let stop = ExternalStop::default();
    let records = Arc::new(Mutex::new(Vec::new()));
    let captured = records.clone();
    let requested = stop.clone();
    let mut opts = options(20);
    opts.external_stop = Some(stop);
    opts.logger = Some(SrLogger::with_interval(
        move |payload| {
            captured.lock().unwrap().push(payload.cycles_remaining.clone());
            requested.request();
        },
        1,
    ));
    equation_search(&[dataset()], &opts).unwrap();
    assert_eq!(*records.lock().unwrap(), [vec![19]]);
}
