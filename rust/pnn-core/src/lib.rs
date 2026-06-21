pub mod diagnostics;
pub mod inference;
pub mod knn;
pub(crate) mod math;
pub mod model;
pub mod predict;
pub mod precompute;
pub mod types;

pub use diagnostics::{build_diagnostics, DiagnosticsOutput};
pub use inference::sample_posterior;
pub use model::{log_joint, log_likelihood, log_prior_beta, log_prior_k, log_sum_exp};
pub use predict::{argmax, batch_sorted_neighbors, predict_proba, softmax_stable};
pub use precompute::{build_count_tensor, pairwise_sq_distances};
pub use types::{
    CountTensor, DataMatrix, InferenceMethod, Labels, ModelError, PnnModel, PosteriorDraw,
    SamplerConfig, SamplerResult,
};
