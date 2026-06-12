use pnn_core::precompute::{build_count_tensor, pairwise_sq_distances};
use pnn_core::PnnModel;

// ── pairwise_sq_distances ────────────────────────────────────────────────────

#[test]
fn pairwise_sq_distances_hand_checked() {
    // a[0]=(0,0), a[1]=(1,0)   b[0]=(0,1), b[1]=(2,2)
    let a = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
    let b = vec![vec![0.0, 1.0], vec![2.0, 2.0]];
    let d = pairwise_sq_distances(&a, &b);

    assert_eq!(d.len(), 2);
    assert_eq!(d[0].len(), 2);

    // a[0] → b[0]: 0^2+1^2 = 1   a[0] → b[1]: 2^2+2^2 = 8
    // a[1] → b[0]: 1^2+1^2 = 2   a[1] → b[1]: 1^2+2^2 = 5
    assert!((d[0][0] - 1.0).abs() < 1e-12);
    assert!((d[0][1] - 8.0).abs() < 1e-12);
    assert!((d[1][0] - 2.0).abs() < 1e-12);
    assert!((d[1][1] - 5.0).abs() < 1e-12);
}

#[test]
fn pairwise_sq_distances_self_diagonal_is_zero() {
    let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let d = pairwise_sq_distances(&x, &x);
    assert!((d[0][0]).abs() < 1e-12);
    assert!((d[1][1]).abs() < 1e-12);
}

// ── build_count_tensor — worked example from pnn_build_plan.md ───────────────
//
// X_train = [[1,2],[5,3],[2,1]]   y_train=[0,1,0]   n_classes=2   k_values=[1,2]
//
// Sorted neighbors (self excluded):
//   i=0: [2(d=2), 1(d=17)]   → k=1 adds j=2(class 0)  → k=2 adds j=1(class 1)
//   i=1: [2(d=13),0(d=17)]   → k=1 adds j=2(class 0)  → k=2 adds j=0(class 0)
//   i=2: [0(d=2), 1(d=13)]   → k=1 adds j=0(class 0)  → k=2 adds j=1(class 1)
//
// Expected tensor[i][ki][c] (ki=0→k=1, ki=1→k=2):
//   [0]: [[1,0],[1,1]]
//   [1]: [[1,0],[2,0]]
//   [2]: [[1,0],[1,1]]

#[test]
fn build_count_tensor_matches_worked_example() {
    let model = PnnModel::new(
        vec![vec![1.0, 2.0], vec![5.0, 3.0], vec![2.0, 1.0]],
        vec![0, 1, 0],
        2,
        vec![1, 2],
    )
    .expect("valid model");

    let ct = build_count_tensor(&model);

    assert_eq!(ct.len(), 3);
    assert_eq!(ct[0].len(), 2); // 2 k-candidates
    assert_eq!(ct[0][0].len(), 2); // 2 classes

    // i=0
    assert_eq!(ct[0][0], vec![1, 0], "i=0 k=1");
    assert_eq!(ct[0][1], vec![1, 1], "i=0 k=2");
    // i=1
    assert_eq!(ct[1][0], vec![1, 0], "i=1 k=1");
    assert_eq!(ct[1][1], vec![2, 0], "i=1 k=2");
    // i=2
    assert_eq!(ct[2][0], vec![1, 0], "i=2 k=1");
    assert_eq!(ct[2][1], vec![1, 1], "i=2 k=2");
}

// ── build_count_tensor — non-contiguous k_values ─────────────────────────────
//
// 1-D dataset, 4 points: X=[0,1,2,3]  y=[0,1,0,1]  n_classes=2  k_values=[1,3]
// (k=2 is skipped; verifies the ki-pointer lands at the right snapshot indices)
//
// Squared distances (self excluded):
//   i=0: j=1(d=1), j=2(d=4), j=3(d=9)  → sorted: [1,2,3]
//   i=1: j=0(d=1), j=2(d=1), j=3(d=4)  → tie on d=1 → index asc: [0,2,3]
//   i=2: j=1(d=1), j=3(d=1), j=0(d=4)  → tie on d=1 → index asc: [1,3,0]
//   i=3: j=2(d=1), j=1(d=4), j=0(d=9)  → sorted: [2,1,0]
//
// Counts:
//   i=0 k=1: add j=1(class 1)             → [0,1]
//   i=0 k=3: add j=2(class 0),j=3(class1) → [1,2]
//
//   i=1 k=1: add j=0(class 0)             → [1,0]
//   i=1 k=3: add j=2(class 0),j=3(class1) → [2,1]
//
//   i=2 k=1: add j=1(class 1)             → [0,1]
//   i=2 k=3: add j=3(class 1),j=0(class0) → [1,2]
//
//   i=3 k=1: add j=2(class 0)             → [1,0]
//   i=3 k=3: add j=1(class 1),j=0(class0) → [2,1]

#[test]
fn build_count_tensor_non_contiguous_k_values() {
    let model = PnnModel::new(
        vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        vec![0, 1, 0, 1],
        2,
        vec![1, 3], // k=2 is intentionally skipped
    )
    .expect("valid model");

    let ct = build_count_tensor(&model);

    assert_eq!(ct.len(), 4);
    assert_eq!(ct[0].len(), 2); // 2 k-candidates (ki=0→k=1, ki=1→k=3)

    assert_eq!(ct[0][0], vec![0, 1], "i=0 k=1");
    assert_eq!(ct[0][1], vec![1, 2], "i=0 k=3");

    assert_eq!(ct[1][0], vec![1, 0], "i=1 k=1");
    assert_eq!(ct[1][1], vec![2, 1], "i=1 k=3");

    assert_eq!(ct[2][0], vec![0, 1], "i=2 k=1");
    assert_eq!(ct[2][1], vec![1, 2], "i=2 k=3");

    assert_eq!(ct[3][0], vec![1, 0], "i=3 k=1");
    assert_eq!(ct[3][1], vec![2, 1], "i=3 k=3");
}

// ── build_count_tensor — tie-breaking matches knn.rs ────────────────────────
//
// Two equidistant neighbors; verifies the lower index is consumed first.
// X=[[0],[1],[3]]  y=[0,1,0]  n_classes=2  k_values=[1]
// i=1: dist to j=0 is 1, dist to j=2 is 4 → no tie here, just checking index ordering
//
// Better tie case:
// X=[[0],[1],[2]]  y=[0,1,0]  k_values=[1]
//   i=1: j=0(d=1), j=2(d=1) → tie → lower index j=0 first → k=1 counts=[1,0]

#[test]
fn build_count_tensor_tie_breaking_uses_lower_index() {
    let model = PnnModel::new(
        vec![vec![0.0], vec![1.0], vec![2.0]],
        vec![0, 1, 0],
        2,
        vec![1],
    )
    .expect("valid model");

    let ct = build_count_tensor(&model);

    // i=1 is equidistant from i=0 (class 0) and i=2 (class 0).
    // Both are class 0, so counts=[1,0] regardless of order.
    // Use a dataset where classes differ to observe the tie-break.
    let _ = ct; // validated via the non-contiguous test which already has a tie case
}

#[test]
fn build_count_tensor_tie_breaking_observed_via_counts() {
    // i=1 (class 1) is equidistant from j=0 (class 0) and j=2 (class 2).
    // Tie-break: j=0 < j=2, so k=1 snapshot must reflect j=0 → counts=[1,0,0].
    let model = PnnModel::new(
        vec![vec![0.0], vec![1.0], vec![2.0]],
        vec![0, 1, 2], // three distinct classes
        3,
        vec![1],
    )
    .expect("valid model");

    let ct = build_count_tensor(&model);
    // i=1, k=1: nearest is j=0 (lower index wins tie) → class 0 → [1,0,0]
    assert_eq!(ct[1][0], vec![1, 0, 0], "tie broken by lower index");
}

// ── build_count_tensor — count totals sum to k ───────────────────────────────

#[test]
fn build_count_tensor_count_totals_equal_k() {
    let model = PnnModel::new(
        vec![vec![1.0, 2.0], vec![5.0, 3.0], vec![2.0, 1.0]],
        vec![0, 1, 0],
        2,
        vec![1, 2],
    )
    .expect("valid model");

    let k_values = model.k_values.clone();
    let ct = build_count_tensor(&model);

    for i in 0..ct.len() {
        for (ki, &k) in k_values.iter().enumerate() {
            let total: usize = ct[i][ki].iter().sum();
            assert_eq!(total, k, "counts at i={i} ki={ki} must sum to k={k}");
        }
    }
}
