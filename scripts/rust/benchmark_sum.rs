use std::time::Instant;
use std::hint::black_box;
use rand::Rng;

// 1. The Iterator-based approach
fn squared_euclidean_distance_iter(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let delta = x - y;
            delta * delta
        })
        .sum()
}

// 2. The For-loop approach
fn squared_euclidean_distance_for(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let min_len = a.len().min(b.len());
    
    for i in 0..min_len {
        let delta = a[i] - b[i];
        sum += delta * delta;
    }
    
    sum
}
fn generate_mock_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    // Start a local random number generator
    let mut rng = rand::thread_rng();

    //generate two vectors of the specified size with random floating-point numbers
    // let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
    // let b: Vec<f64> = (0..size).map(|i| i as f64 * 0.2).collect();
    let a: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..1000.0)).collect();
    let b: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..1000.0)).collect();
    (a, b)
}

fn benchmark(a: &[f64], b: &[f64]) {
    // Warm-up to ensure CPU caches and scaling don't skew the first run
    let _ = squared_euclidean_distance_iter(&a, &b);
    let _ = squared_euclidean_distance_for(&a, &b);

    println!("Running benchmark...");

    // Benchmark Iterator
    let start_iter = Instant::now();
    // black_box prevents the compiler from pre-calculating the result at compile time
    let res_iter = squared_euclidean_distance_iter(black_box(&a), black_box(&b));
    let duration_iter = start_iter.elapsed();

    // Benchmark For Loop
    let start_for = Instant::now();
    let res_for = squared_euclidean_distance_for(black_box(&a), black_box(&b));
    let duration_for = start_for.elapsed();

    println!("Result (Iterator) : {}", res_iter);
    println!("Result (For Loop) : {}\n", res_for);
    
    println!("⏱️ Iterator Time : {:?}", duration_iter);
    println!("⏱️ For Loop Time : {:?}", duration_for);
}
fn main() {
    println!("Generating mock data (10 million dimensions)...");
    let sizes = Vec::from([10_000, 100_000, 1_000_000, 10_000_000]);

    for size in sizes {
        println!("\n📊 Benchmarking with vector size: {}", size);
        let (a, b) = generate_mock_data(size);
         benchmark(&a, &b);
    }

    println!("\n⚠️ Note: If you didn't run this with `--release`, the results are deceiving!");
}
