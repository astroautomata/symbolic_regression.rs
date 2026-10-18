#[cfg(feature = "progress")]
mod imp {
    use std::fmt::Display;
    use std::io::IsTerminal;
    use std::time::{Duration, Instant};

    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::strings::StringTreeOptions;
    use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
    use num_traits::Float;

    use crate::dataset::Dataset;
    use crate::hall_of_fame::HallOfFame;
    use crate::options::{LossScale, Options, OutputStyle};
    use crate::units::{UnitLabels, Units};

    struct ProgressTracking {
        last_speed_time: Instant,
        evals_last: u64,
        speeds: Vec<f64>,
        last_msg_update: Instant,
    }

    #[derive(Copy, Clone, Debug)]
    pub(crate) struct RenderOptions {
        pub(crate) ansi: bool,
        /// `LossScale::Log` adds a Score column, as in SymbolicRegression.jl.
        pub(crate) loss_scale: LossScale,
    }

    pub(crate) struct SearchProgress {
        show: bool,
        print: bool,
        total_cycles: usize,
        populations: usize,
        bar: ProgressBar,
        start_time: Instant,
        msg_min_interval: Duration,
        tracking: ProgressTracking,
        render: RenderOptions,
    }

    impl SearchProgress {
        /// `show` is false for a multi-output search, which SymbolicRegression.jl shows no bar for.
        pub(crate) fn new<T: Float, Ops, const D: usize>(
            options: &Options<T, Ops, D>,
            show: bool,
            total_cycles: usize,
        ) -> Self {
            let show =
                show && options.progress && options.verbosity.unwrap_or(1) > 0 && std::io::stderr().is_terminal();

            let ansi = match options.output_style {
                OutputStyle::Plain => {
                    console::set_colors_enabled_stderr(false);
                    false
                }
                OutputStyle::Ansi => {
                    console::set_colors_enabled_stderr(true);
                    true
                }
                OutputStyle::Auto => console::colors_enabled_stderr(),
            };
            let render = RenderOptions {
                ansi,
                loss_scale: options.loss_scale,
            };

            let bar = if show {
                let pb = ProgressBar::new(total_cycles as u64);
                pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
                pb.set_prefix(format!("Evolving for {} iterations...", options.niterations));
                let style = ProgressStyle::with_template(
                    "{prefix} {wide_bar} {pos:>7}/{len:7} [{elapsed_precise}<{eta_precise}]\n{msg}",
                )
                .unwrap();
                pb.set_style(style);
                pb
            } else {
                ProgressBar::hidden()
            };

            Self {
                show,
                print: !options.progress && options.verbosity.unwrap_or(1) > 0,
                total_cycles,
                populations: options.populations,
                bar,
                start_time: Instant::now(),
                msg_min_interval: Duration::from_millis(500),
                tracking: ProgressTracking {
                    last_speed_time: Instant::now(),
                    evals_last: 0,
                    speeds: Vec::new(),
                    last_msg_update: Instant::now(),
                },
                render,
            }
        }

        pub(crate) fn warn(&self, message: &str) {
            if self.show {
                self.bar.println(format!("Warning: {message}"));
            }
        }

        pub(crate) fn set_initial_evals(&mut self, total_evals: u64) {
            self.tracking.evals_last = total_evals;
            self.tracking.last_speed_time = Instant::now();
        }

        pub(crate) fn on_cycle_complete<T, Ops, const D: usize>(
            &mut self,
            halls: &[HallOfFame<T, Ops, D>],
            datasets: &[Dataset<T>],
            options: &Options<T, Ops, D>,
            total_evals: u64,
            cycles_remaining: usize,
            print_precision: usize,
        ) -> Result<(), String>
        where
            T: Float + num_traits::ToPrimitive + Display,
            Ops: OperatorSet,
        {
            if self.print
                && (self.tracking.last_msg_update.elapsed() >= Duration::from_secs(5) || cycles_remaining == 0)
            {
                let state = crate::printing::format_search_state(self.total_cycles, cycles_remaining, self.populations);
                println!("{state}");
                for (j, (hall, dataset)) in halls.iter().zip(datasets).enumerate() {
                    if datasets.len() > 1 {
                        println!("Best equations for output {}", j + 1);
                    }
                    let hall = format_hall_of_fame(
                        hall,
                        options,
                        options.terminal_width.unwrap_or(100).max(100),
                        usize::MAX,
                        self.render,
                        print_precision,
                        dataset.units.as_ref(),
                        Some(&dataset.display_variable_names),
                        &dataset.y_variable_name,
                    )?;
                    println!("{hall}");
                }
                self.tracking.last_msg_update = Instant::now();
            }
            if !self.show {
                return Ok(());
            }
            self.bar.inc(1);
            update_progress_msg(ProgressMsgCtx {
                pb: &self.bar,
                hall: &halls[0],
                options,
                start_time: self.start_time,
                msg_min_interval: self.msg_min_interval,
                progress: &mut self.tracking,
                render: self.render,
                total_evals,
                cycles_remaining,
                print_precision,
                dataset: &datasets[0],
            })
        }

        pub(crate) fn finish(&self) {
            if self.show {
                self.bar.finish();
            }
        }
    }

    struct ProgressMsgCtx<'a, T: Float, Ops, const D: usize> {
        pb: &'a ProgressBar,
        hall: &'a HallOfFame<T, Ops, D>,
        options: &'a Options<T, Ops, D>,
        start_time: Instant,
        msg_min_interval: Duration,
        progress: &'a mut ProgressTracking,
        render: RenderOptions,
        total_evals: u64,
        cycles_remaining: usize,
        print_precision: usize,
        dataset: &'a Dataset<T>,
    }

    fn update_progress_msg<T, Ops, const D: usize>(ctx: ProgressMsgCtx<'_, T, Ops, D>) -> Result<(), String>
    where
        T: Float + num_traits::ToPrimitive + Display,
        Ops: OperatorSet,
    {
        let ProgressMsgCtx {
            pb,
            hall,
            options,
            start_time,
            msg_min_interval,
            progress,
            render,
            total_evals,
            cycles_remaining,
            print_precision,
            dataset,
        } = ctx;

        let now = Instant::now();
        if now.duration_since(progress.last_speed_time) >= Duration::from_secs(1) {
            let dt = now.duration_since(progress.last_speed_time).as_secs_f64().max(1e-9);
            let evals_since = total_evals.saturating_sub(progress.evals_last);
            progress.speeds.push((evals_since as f64) / dt);
            if progress.speeds.len() > 20 {
                progress.speeds.remove(0);
            }
            progress.evals_last = total_evals;
            progress.last_speed_time = now;
        }

        if now.duration_since(progress.last_msg_update) < msg_min_interval && cycles_remaining != 0 {
            return Ok(());
        }

        let term_width = {
            let (_, w) = console::Term::stderr().size();
            options.terminal_width.unwrap_or(w as usize).max(80)
        };

        let avg_speed = if progress.speeds.is_empty() {
            None
        } else {
            Some(progress.speeds.iter().copied().sum::<f64>() / (progress.speeds.len() as f64))
        };

        let info_line = match avg_speed {
            Some(s) => format!(
                "Info: eval/s~{:<8.2e} | evals={} | t={:.1}s",
                s,
                total_evals,
                start_time.elapsed().as_secs_f64()
            ),
            None => format!(
                "Info: eval/s=[.....]  | evals={} | t={:.1}s",
                total_evals,
                start_time.elapsed().as_secs_f64()
            ),
        };

        let hof = format_hall_of_fame(
            hall,
            options,
            term_width,
            usize::MAX,
            render,
            print_precision,
            dataset.units.as_ref(),
            Some(&dataset.display_variable_names),
            &dataset.y_variable_name,
        )?;
        let hof_title = "Hall of Fame:".to_string();
        pb.set_message(format!(
            "{}\n{}\n{}",
            truncate_to_width(&info_line, term_width, render),
            truncate_to_width(&hof_title, term_width, render),
            hof
        ));

        progress.last_msg_update = now;
        Ok(())
    }

    /// SymbolicRegression.jl's hall of fame. Each equation reads `y = ...`; with units,
    /// `y[kg] = x1[m] * 3.2[?]`: features carry their units, constants whose units are free carry
    /// `[?]`, and the target carries its units, or `[?]` when only the features have units.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn format_hall_of_fame<T, Ops, const D: usize>(
        hall: &HallOfFame<T, Ops, D>,
        options: &Options<T, Ops, D>,
        terminal_width: usize,
        max_entries: usize,
        render: RenderOptions,
        print_precision: usize,
        units: Option<&Units>,
        variable_names: Option<&[String]>,
        y_variable_name: &str,
    ) -> Result<String, String>
    where
        T: Float + num_traits::ToPrimitive + Display,
        Ops: OperatorSet,
    {
        let terminal_width = terminal_width.max(80);
        let border = "─".repeat(terminal_width.saturating_sub(1));
        let show_score = render.loss_scale == LossScale::Log;
        let style = console::Style::new().bold().underlined();
        let mut header = String::new();
        for name in ["Complexity", "Loss", "Score"] {
            if name == "Score" && !show_score {
                continue;
            }
            let styled = if render.ansi {
                style.apply_to(name).to_string()
            } else {
                name.to_string()
            };
            header.push_str(&format!("{styled}{}  ", " ".repeat(10 - name.len())));
        }
        header.push_str(&if render.ansi {
            style.apply_to("Equation").to_string()
        } else {
            "Equation".to_string()
        });

        let mut out = String::new();
        out.push_str(&border);
        out.push('\n');
        out.push_str(&truncate_to_width(&header, terminal_width, render));
        out.push('\n');

        let scores = hall.pareto_scores(render.loss_scale)?;
        let labels = units.map(|u| UnitLabels::new(u, options.dimensionless_constants_only));
        let prefix = format!(
            "{y_variable_name}{} = ",
            labels.as_ref().map_or("", |l| l.target.as_str())
        );
        for (m, score) in hall.pareto_front().into_iter().zip(scores).take(max_entries) {
            let loss = m.loss.to_f64().unwrap_or(f64::INFINITY);
            let mut stats = format!("{:<10}  {:<10.3e}  ", m.complexity, loss);
            if show_score {
                stats.push_str(&format!("{:<10.3e}  ", score.to_f64().unwrap_or(f64::NAN)));
            }
            let left_cols_width = stats.chars().count();

            let eqn_plain = prefix.clone()
                + &m.expr.string(
                    options,
                    StringTreeOptions {
                        print_precision: Some(print_precision),
                        variable_names,
                        pretty: true,
                        variable_suffixes: labels.as_ref().map(|l| l.variables.as_slice()),
                        constant_suffix: labels.as_ref().map_or("", |l| l.constant),
                        ..Default::default()
                    },
                );
            let eqn_lines = wrap_equation(&eqn_plain, terminal_width, left_cols_width);

            if let Some((first, rest)) = eqn_lines.split_first() {
                let first_eq = first.to_string();
                out.push_str(&truncate_to_width(&(stats.clone() + &first_eq), terminal_width, render));
                out.push('\n');
                for line in rest {
                    let eq = line.to_string();
                    let padded = eq;
                    out.push_str(&truncate_to_width(&padded, terminal_width, render));
                    out.push('\n');
                }
            } else {
                out.push_str(&truncate_to_width(&stats, terminal_width, render));
                out.push('\n');
            }
        }

        out.push_str(&border);
        Ok(out)
    }

    fn wrap_equation(eqn: &str, terminal_width: usize, left_cols_width: usize) -> Vec<String> {
        crate::printing::wrap_equation_string(eqn, left_cols_width, terminal_width)
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn truncate_to_width(s: &str, terminal_width: usize, render: RenderOptions) -> String {
        let max = terminal_width.saturating_sub(1);
        if render.ansi {
            return console::truncate_str(s, max, "").into_owned();
        }
        if s.chars().count() <= max {
            s.to_string()
        } else {
            s.chars().take(max).collect()
        }
    }
}

#[cfg(not(feature = "progress"))]
mod imp {
    use num_traits::Float;

    use crate::dataset::Dataset;
    use crate::hall_of_fame::HallOfFame;
    use crate::options::Options;

    pub(crate) struct SearchProgress;

    impl SearchProgress {
        pub(crate) fn new<T: Float, Ops, const D: usize>(
            _options: &Options<T, Ops, D>,
            _show: bool,
            _total_cycles: usize,
        ) -> Self {
            Self
        }

        pub(crate) fn warn(&self, _message: &str) {}

        pub(crate) fn set_initial_evals(&mut self, _total_evals: u64) {}

        pub(crate) fn on_cycle_complete<T: Float, Ops, const D: usize>(
            &mut self,
            _halls: &[HallOfFame<T, Ops, D>],
            _datasets: &[Dataset<T>],
            _options: &Options<T, Ops, D>,
            _total_evals: u64,
            _cycles_remaining: usize,
            _print_precision: usize,
        ) -> Result<(), String> {
            Ok(())
        }

        pub(crate) fn finish(&self) {}
    }
}

pub(crate) use imp::SearchProgress;
#[cfg(all(test, feature = "progress"))]
pub(crate) use imp::{RenderOptions, format_hall_of_fame};
