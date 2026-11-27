//! Admission decisions based on a fresh device-memory measurement.

/// Required device memory in bytes for the selected, unchanged configuration.
#[derive(Clone, Copy, Debug)]
pub struct VramRequest {
    pub weights: u64,
    pub experts: u64,
    pub kv_cache: u64,
    pub activations: u64,
    pub cuda_workspace: u64,
    pub runtime: u64,
    pub reserve: u64,
}

/// An unavailable measurement, unrepresentable request, or insufficient capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    Unavailable,
    Overflow,
    Insufficient { required: u64, available: u64 },
}

/// Checks the complete budget without changing the requested model or context.
pub fn admit_vram(free: Option<u64>, request: VramRequest) -> Result<u64, AdmissionError> {
    let available = free.ok_or(AdmissionError::Unavailable)?;
    let required = [
        request.weights,
        request.experts,
        request.kv_cache,
        request.activations,
        request.cuda_workspace,
        request.runtime,
        request.reserve,
    ]
    .into_iter()
    .try_fold(0_u64, u64::checked_add)
    .ok_or(AdmissionError::Overflow)?;
    available
        .checked_sub(required)
        .ok_or(AdmissionError::Insufficient {
            required,
            available,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_checks_missing_measurement_overflow_and_capacity() {
        let request = VramRequest {
            weights: 2,
            experts: 3,
            kv_cache: 5,
            activations: 7,
            cuda_workspace: 11,
            runtime: 13,
            reserve: 17,
        };
        assert_eq!(admit_vram(Some(58), request), Ok(0));
        assert_eq!(
            admit_vram(Some(57), request),
            Err(AdmissionError::Insufficient {
                required: 58,
                available: 57
            })
        );
        assert_eq!(admit_vram(None, request), Err(AdmissionError::Unavailable));
        assert_eq!(
            admit_vram(
                Some(u64::MAX),
                VramRequest {
                    weights: u64::MAX,
                    ..request
                }
            ),
            Err(AdmissionError::Overflow)
        );
    }
}
