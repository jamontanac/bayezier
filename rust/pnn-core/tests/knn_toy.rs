use pnn_core::knn::k_nearest;

#[test]
fn returns_expected_indices_for_hand_checked_five_point_example() {
    let data = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ];
    let query = [0.9, 0.8];

    let neighbors = k_nearest(&data, &query, 3).expect("k_nearest should succeed");

    assert_eq!(neighbors, vec![3, 1, 2]);
}

#[test]
fn breaks_distance_ties_by_index_order() {
    let data = vec![
        vec![0.0, 1.0],
        vec![0.0, -1.0],
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
    ];
    let query = [0.0, 0.0];

    let neighbors = k_nearest(&data, &query, 2).expect("k_nearest should succeed");

    assert_eq!(neighbors, vec![0, 1]);
}
