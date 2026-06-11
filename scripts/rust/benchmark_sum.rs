use std::time::Instant;
use std::hint::black_box;
use std::cmp::Ordering;
use rand::Rng;

// --- Foundation Type Definitions for Standalone Operation ---
type DataMatrix = Vec<Vec<f64>>;

#[derive(Debug)]
pub enum KnnError {
    DimensionMismatch,
    InvalidK,
}
impl std::fmt::Display for KnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for KnnError {}

// Helper validation logic to fulfill your function's signature
fn validate_inputs(data: &DataMatrix, query: &[f64], k: usize) -> Result<(), KnnError> {
    if data.is_empty() || query.is_empty() { return Err(KnnError::DimensionMismatch); }
    if k == 0 || k > data.len() { return Err(KnnError::InvalidK); }
    Ok(())
}

// The Iterator-based approach
fn squared_euclidean_distance_iter(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let delta = x - y;
            delta * delta
        })
        .sum()
}

//
fn assert_result_parity(run_idx: usize, size_context: usize, res_iter: &[usize], res_for: &[usize]) {
    if res_iter != res_for {
        eprintln!(
            "\n❌ PARITY VIOLATION DETECTED! [Size Context: {}, Realization Run: {}]",
            size_context, run_idx
        );
        eprintln!("Iterator Output : {:?}", res_iter);
        eprintln!("For-Loop Output : {:?}", res_for);
        
        // Find the exact element mismatch index for easier debugging
        let mismatch_pos = res_iter.iter().zip(res_for.iter()).position(|(a, b)| a != b);
        if let Some(pos) = mismatch_pos {
            eprintln!("First mismatch occurs at top-k index item [{}]: {} != {}", pos, res_iter[pos], res_for[pos]);
        }
        
        panic!("Execution halted due to strict consistency failure.");
    }
}


// The For-loop approach
fn squared_euclidean_distance_for(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let min_len = a.len().min(b.len());
    
    for i in 0..min_len {
        let delta = a[i] - b[i];
        sum += delta * delta;
    }
    
    sum
}

pub fn k_nearest_iter(data: &DataMatrix, query: &[f64], k: usize) -> Result<Vec<usize>, KnnError> {
    validate_inputs(data, query, k)?;

    let mut ranked: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(idx, row)| (idx, squared_euclidean_distance_for(row, query)))
        .collect();

    if k > 0 && k <= ranked.len() {
        ranked.select_nth_unstable_by(k - 1, |(idx_a, dist_a), (idx_b, dist_b)| {
            dist_a.partial_cmp(dist_b).unwrap_or(Ordering::Equal).then_with(|| idx_a.cmp(idx_b))
        });
    }

    let mut top_k = ranked;
    top_k.truncate(k);
    top_k.sort_by(|(idx_a, dist_a), (idx_b, dist_b)| {
        dist_a.partial_cmp(dist_b).unwrap_or(Ordering::Equal).then_with(|| idx_a.cmp(idx_b))
    });

    Ok(top_k.into_iter().map(|(idx, _)| idx).collect())
}

pub fn k_nearest_for(data: &DataMatrix, query: &[f64], k: usize) -> Result<Vec<usize>, KnnError> {
    validate_inputs(data, query, k)?;

    let mut ranked: Vec<(usize, f64)> = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let dist = squared_euclidean_distance_for(&data[i], query);
        ranked.push((i, dist));
    }

    for i in 0..k {
        let mut min_idx = i;
        
        for j in (i + 1)..ranked.len() {
            let dist_j = ranked[j].1;
            let dist_min = ranked[min_idx].1;
            
            let cmp = dist_j.partial_cmp(&dist_min).unwrap_or(Ordering::Equal)
                .then_with(|| ranked[j].0.cmp(&ranked[min_idx].0));
                
            if cmp == Ordering::Less {
                min_idx = j;
            }
        }
        
        if min_idx != i {
            ranked.swap(i, min_idx);
        }
    }

    let mut result = Vec::with_capacity(k);
    for i in 0..k {
        result.push(ranked[i].0);
    }

    Ok(result)
}

fn generate_mock_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let a: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..1000.0)).collect();
    let b: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..1000.0)).collect();
    (a, b)
}

fn generate_mock_matrix(rows: usize, cols: usize) -> (DataMatrix, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut matrix = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row: Vec<f64> = (0..cols).map(|_| rng.gen_range(0.0..1000.0)).collect();
        matrix.push(row);
    }
    let query: Vec<f64> = (0..cols).map(|_| rng.gen_range(0.0..1000.0)).collect();
    (matrix, query)
}

// --- New Core Statistical Analysis Utility ---
fn calculate_stats(durations_nanos: &[f64]) -> (f64, f64) {
    let n = durations_nanos.len() as f64;
    if n == 0.0 { return (0.0, 0.0); }
    
    let sum: f64 = durations_nanos.iter().sum();
    let mean = sum / n;
    
    let variance: f64 = durations_nanos
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>() / (n - 1.0).max(1.0); // Sample variance (bessel correction)
        
    let std_dev = variance.sqrt();
    (mean, std_dev)
}

fn format_duration_metric(nanos: f64) -> String {
    if nanos >= 1_000_000_000.0 {
        format!("{:.3} s", nanos / 1_000_000_000.0)
    } else if nanos >= 1_000_000.0 {
        format!("{:.3} ms", nanos / 1_000_000.0)
    } else if nanos >= 1_000.0 {
        format!("{:.3} µs", nanos / 1_000.0)
    } else {
        format!("{:.3} ns", nanos)
    }
}
fn benchmark_distance(a: &[f64], b: &[f64], runs: usize) {
    // Warm-up
    let _ = squared_euclidean_distance_iter(&a, &b);
    let _ = squared_euclidean_distance_for(&a, &b);

    let mut iter_times = Vec::with_capacity(runs);
    let mut for_times = Vec::with_capacity(runs);

    // Run Iterator Approached Realizations
    for _ in 0..runs {
        let start = Instant::now();
        let _res = squared_euclidean_distance_iter(black_box(a), black_box(b));
        iter_times.push(start.elapsed().as_nanos() as f64);
    }

    // Run For-Loop Approached Realizations
    for _ in 0..runs {
        let start = Instant::now();
        let _res = squared_euclidean_distance_for(black_box(a), black_box(b));
        for_times.push(start.elapsed().as_nanos() as f64);
    }

    let (iter_mean, iter_std) = calculate_stats(&iter_times);
    let (for_mean, for_std) = calculate_stats(&for_times);

    println!("  ⏱️ Iterator : {} ± {}", format_duration_metric(iter_mean), format_duration_metric(iter_std));
    println!("  ⏱️ For Loop : {} ± {}", format_duration_metric(for_mean), format_duration_metric(for_std));
}

fn benchmark_knn(data: &DataMatrix, query: &[f64], k: usize, runs: usize) {
    // Warm-up
    let _ = k_nearest_iter(data, query, k);
    let _ = k_nearest_for(data, query, k);

    let mut iter_times = Vec::with_capacity(runs);
    let mut for_times = Vec::with_capacity(runs);

    // Run Iterator KNN Realizations
    for run in 0..runs {
        let start = Instant::now();
        let res_iter = k_nearest_iter(black_box(data), black_box(query), black_box(k)).unwrap();
        iter_times.push(start.elapsed().as_nanos() as f64);

        // Run For Loop KNN Realizations directly inside or side-by-side
        let start_for = Instant::now();
        let res_for = k_nearest_for(black_box(data), black_box(query), black_box(k)).unwrap();
        for_times.push(start_for.elapsed().as_nanos() as f64);

        // Execute strict differential validation immediately on the arrays
        assert_result_parity(run, data.len(), &res_iter, &res_for);
    }

    let (iter_mean, iter_std) = calculate_stats(&iter_times);
    let (for_mean, for_std) = calculate_stats(&for_times);

    println!("  ⏱️ Iterator KNN : {} ± {}", format_duration_metric(iter_mean), format_duration_metric(iter_std));
    println!("  ⏱️ For Loop KNN : {} ± {}", format_duration_metric(for_mean), format_duration_metric(for_std));
}
// fn benchmark_knn(data: &DataMatrix, query: &[f64], k: usize, runs: usize) {
//     // Warm-up
//     let _ = k_nearest_iter(data, query, k);
//     let _ = k_nearest_for(data, query, k);
//
//     let mut iter_times = Vec::with_capacity(runs);
//     let mut for_times = Vec::with_capacity(runs);
//
//     // Run Iterator KNN Realizations
//     for _ in 0..runs {
//         let start = Instant::now();
//         let _res = k_nearest_iter(black_box(data), black_box(query), black_box(k)).unwrap();
//         iter_times.push(start.elapsed().as_nanos() as f64);
//     }
//
//     // Run For Loop KNN Realizations
//     for _ in 0..runs {
//         let start = Instant::now();
//         let _res = k_nearest_for(black_box(data), black_box(query), black_box(k)).unwrap();
//         for_times.push(start.elapsed().as_nanos() as f64);
//     }
//
//     let (iter_mean, iter_std) = calculate_stats(&iter_times);
//     let (for_mean, for_std) = calculate_stats(&for_times);
//
//     println!("  ⏱️ Iterator KNN : {} ± {}", format_duration_metric(iter_mean), format_duration_metric(iter_std));
//     println!("  ⏱️ For Loop KNN : {} ± {}", format_duration_metric(for_mean), format_duration_metric(for_std));
// }

fn main() {
    const RUNS: usize = 30; // Number of realizations per statistical benchmark block
    println!("Starting robust multi-run benchmark (Realizations per test: {})...", RUNS);
    
    let sizes = vec![10_000, 100_000, 1_000_000, 10_000_000];
    for size in sizes {
        println!("\n📊 Vector Distance Benchmark [Dimensions: {}]", size);
        let (a, b) = generate_mock_data(size);
        benchmark_distance(&a, &b, RUNS);
    }
    
    let sizes_knn = vec![1000, 10_000, 50_000, 100_000, 1_000_000]; // Trimmed the 1M matrix slightly to avoid long test waits
    for size in sizes_knn {
        println!("\n📊 KNN Model Benchmark [Matrix: {} x 1000]", size);
        let (matrix, query) = generate_mock_matrix(size, 1000);
        benchmark_knn(&matrix, &query, 10, RUNS);
    }

    println!("\n⚠️ Note: Always run with `cargo run --release` to ensure true measurements!");
}
