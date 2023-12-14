use nalgebra::base::dimension::Const;
use nalgebra::base::{ArrayStorage, Storage};
use nalgebra::storage::Owned;
use nalgebra::{Matrix, RawStorageMut, RealField, Vector};

use num_traits::cast::NumCast;

use core::default::Default;
use core::marker::Copy;

use crate::base::types::IntermediateStateStateMapping;
use crate::base::types::KalmanState;
use crate::base::types::NonlinearPredictWorkspace;
use crate::base::types::NonlinearProcessModel;

pub trait IntermediateStateSize<T: RealField> {
    type IntermediateState: Sized;
}

pub trait NonlinearPredict<T, const S: usize>: KalmanState<T, S>
where
    T: RealField + NumCast + Copy + Default,
{
    fn predict<const I: usize, PM, ST, W>(
        &mut self,
        process_model: &PM,
        transition: &ST,
        workspace: &mut W,
        dt: T,
    ) where
        ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>,
        PM: NonlinearProcessModel<T, I, S>,
        ST: IntermediateStateStateMapping<T, I, S>,
        W: NonlinearPredictWorkspace<T, S>,
    {
        let (state, cov) = self.state_cov();

        // TODO: Probably less error-prone to zero out the combined transition jacobian and combined_process_before returning it from workspace
        let (combined_transition_jacobian, combined_process_noise) = workspace.workspace_temps();

        let mut intermediate_state = transition.to_process(state);
        process_model.f(&mut intermediate_state, dt);
        let mut intermediate_jacobian = transition.jacobian_matrix(combined_transition_jacobian);
        process_model.transition_jacobian(&intermediate_state, &mut intermediate_jacobian, dt);
        let mut intermediate_noise = transition.noise_matrix(combined_process_noise);
        process_model.process_noise(&mut intermediate_noise, dt);

        *cov = *combined_transition_jacobian * *cov * (*combined_transition_jacobian).transpose()
            + *combined_process_noise;
    }
}

pub trait NonlinearMeasurementModel<const M: usize, const S: usize, T>:
    IntermediateStateSize<T>
where
    T: RealField + NumCast + Copy + Default,
{
    fn initialize_state(&self) -> Self::IntermediateState;

    fn h(
        &self,
        state: &Self::IntermediateState,
        out: &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
    );
    fn measurement_noise(&self) -> Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>;

    fn measurement_jacobian(
        &self,
        state: &Vector<T, Const<S>, Owned<T, Const<S>>>,
        h: &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
    );
}

pub trait NonlinearUpdateWorkspace<const S: usize, const M: usize, T, ISS: IntermediateStateSize<T>>
where
    T: RealField + NumCast + Copy + Default,
{
    fn workspace_temps(
        &mut self,
    ) -> (
        &mut ISS::IntermediateState,
        &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
        &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
        &mut Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
        &mut Matrix<T, Const<S>, Const<M>, ArrayStorage<T, S, M>>,
        &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
        &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
        &mut Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
    );
}

pub trait IntermediateMeasurmentStateMapping<
    const S: usize,
    const M: usize,
    T,
    ISS: IntermediateStateSize<T>,
> where
    T: RealField + NumCast + Copy + Default,
{
    fn to_intermediate(
        &self,
        state: &Vector<T, Const<S>, Owned<T, Const<S>>>,
        out: &mut ISS::IntermediateState,
    );
}

pub struct GenericNonlinearUpdateWorkspace<
    T: RealField + NumCast + Copy + Default,
    const S: usize,
    const M: usize,
    MM: NonlinearMeasurementModel<M, S, T>,
> {
    intermediate_state: MM::IntermediateState,
    predicted_measurement: Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
    temp: Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
    s_: Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
    k: Matrix<T, Const<S>, Const<M>, ArrayStorage<T, S, M>>,
    h: Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
    y: Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
    i: Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
    s_inv: Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
}

impl<
        T: RealField + NumCast + Copy + Default,
        const S: usize,
        const M: usize,
        MM: NonlinearMeasurementModel<M, S, T>,
    > NonlinearUpdateWorkspace<S, M, T, MM> for GenericNonlinearUpdateWorkspace<T, S, M, MM>
{
    fn workspace_temps(
        &mut self,
    ) -> (
        &mut MM::IntermediateState,
        &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
        &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
        &mut Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
        &mut Matrix<T, Const<S>, Const<M>, ArrayStorage<T, S, M>>,
        &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
        &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
        &mut Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
    ) {
        (
            &mut self.intermediate_state,
            &mut self.predicted_measurement,
            &mut self.temp,
            &mut self.s_,
            &mut self.k,
            &mut self.h,
            &mut self.y,
            &mut self.i,
            &mut self.s_inv,
        )
    }
}

impl<
        T: RealField + NumCast + Copy + Default,
        const S: usize,
        const M: usize,
        MM: NonlinearMeasurementModel<M, S, T>,
    > GenericNonlinearUpdateWorkspace<T, S, M, MM>
{
    pub fn new(initial_intermediate_state: MM::IntermediateState) -> Self {
        Self {
            intermediate_state: initial_intermediate_state,
            predicted_measurement: Vector::<T, Const<M>, ArrayStorage<T, M, 1>>::zeros(),
            temp: Matrix::<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>::zeros(),
            s_: Matrix::<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>::zeros(),
            k: Matrix::<T, Const<S>, Const<M>, ArrayStorage<T, S, M>>::zeros(),
            h: Matrix::<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>::zeros(),
            y: Vector::<T, Const<M>, ArrayStorage<T, M, 1>>::zeros(),
            i: Matrix::<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>::identity(),
            s_inv: Matrix::<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>::zeros(),
        }
    }
}

pub trait NonlinearUpdate<T, const S: usize, const M: usize, ISS, Workspace>:
    KalmanState<T, S>
where
    T: RealField + NumCast + Copy + Default,
    ISS: IntermediateStateSize<T>,
    Workspace: NonlinearUpdateWorkspace<S, M, T, ISS>,
{
    fn update<MM, SM>(
        &mut self,
        z: &Vector<T, Const<M>, Owned<T, Const<M>>>,
        measurement_model: &MM,
        mapping: &SM,
        workspace: &mut Workspace,
    ) where
        MM: NonlinearMeasurementModel<M, S, T, IntermediateState = ISS::IntermediateState>,
        SM: IntermediateMeasurmentStateMapping<S, M, T, ISS>,
        ArrayStorage<T, M, S>: Storage<T, Const<M>, Const<S>>,
        ArrayStorage<T, S, S>: Storage<T, Const<S>, Const<S>>,
    {
        let (intermediate_state, predicted_measurement, temp, s_, k, h, y, i, s_inv) =
            workspace.workspace_temps();
        let (state, cov) = self.state_cov();
        mapping.to_intermediate(state, intermediate_state);
        measurement_model.h(intermediate_state, predicted_measurement);

        measurement_model.measurement_jacobian(state, h);
        *temp = *h * *cov;

        *s_ = *temp * h.transpose() + measurement_model.measurement_noise();

        // Compute the Kalman gain using the previously computed S_
        *s_inv = s_.try_inverse().expect("Matrix is not invertible!");
        *k = *cov * h.transpose() * *s_inv;

        // Compute the measurement residual
        *y = z - *predicted_measurement;

        // Update the state estimate
        *state = *state + *k * *y;

        // Update the error covariance matrix using workspace matrices
        *cov = (*i - *k * *h) * *cov;
    }
}

pub struct ExtendedKalmanFilter<T: RealField, const S: usize> {
    x: Vector<T, Const<S>, Owned<T, Const<S>>>,
    p: Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
}

impl<T: RealField + NumCast + Copy + Default, const S: usize> KalmanState<T, S>
    for ExtendedKalmanFilter<T, S>
{
    fn state_cov(
        &mut self,
    ) -> (
        &mut Vector<T, Const<S>, Owned<T, Const<S>>>,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
    ) {
        (&mut self.x, &mut self.p)
    }
}

impl<
        T: RealField + NumCast + Copy + Default,
        const S: usize,
        const M: usize,
        ISS: IntermediateStateSize<T>,
        Workspace: NonlinearUpdateWorkspace<S, M, T, ISS>,
    > NonlinearUpdate<T, S, M, ISS, Workspace> for ExtendedKalmanFilter<T, S>
{
}
impl<T: RealField + NumCast + Copy + Default, const S: usize> NonlinearPredict<T, S>
    for ExtendedKalmanFilter<T, S>
{
}

// This conditionally includes the std library when tests are being run.
#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::types::separate_state_vars::*;
    use crate::base::GenericNonlinearPredictWorkspace;
    use core::marker::PhantomData;
    use nalgebra::{Matrix2, Matrix4, MatrixViewMut, Vector2, Vector4, VectorViewMut};
    use std::println;

    // -- Example of two distinct states in the state space --

    struct GpsMeasurementModel;

    impl<T: RealField + NumCast + Copy + Default> IntermediateStateSize<T> for GpsMeasurementModel {
        type IntermediateState = Vector2<T>;
    }

    impl<T: RealField + NumCast + Copy + Default, const S: usize> NonlinearMeasurementModel<2, S, T>
        for GpsMeasurementModel
    {
        fn initialize_state(&self) -> Self::IntermediateState {
            // Implement the initialization here
            Vector2::<T>::zeros()
        }

        fn h(
            &self,
            state: &Self::IntermediateState,
            out: &mut Vector<T, Const<2>, ArrayStorage<T, 2, 1>>,
        ) {
            out[0] = state[0];
            out[1] = state[1];
        }

        fn measurement_noise(&self) -> Matrix<T, Const<2>, Const<2>, ArrayStorage<T, 2, 2>> {
            Matrix2::new(
                NumCast::from(0.01).unwrap_or_else(T::zero),
                NumCast::from(0.0).unwrap_or_else(T::zero),
                NumCast::from(0.0).unwrap_or_else(T::zero),
                NumCast::from(0.01).unwrap_or_else(T::zero),
            )
        }

        fn measurement_jacobian(
            &self,
            _state: &Vector<T, Const<S>, Owned<T, Const<S>>>,
            h: &mut Matrix<T, Const<2>, Const<S>, ArrayStorage<T, 2, S>>,
        ) {
            // Fill in the Jacobian matrix `H` according to your measurement model's relation with the state
            // Given the example, assuming that only the first two state variables relate to the measurement model:
            h.fill(T::zero());
            h[(0, 0)] = T::one();
            h[(1, 1)] = T::one();
        }
    }

    struct GpsIntermediateMeasurmentStateMapping;

    impl<T: RealField + NumCast + Copy + Default>
        IntermediateMeasurmentStateMapping<4, 2, T, GpsMeasurementModel>
        for GpsIntermediateMeasurmentStateMapping
    {
        fn to_intermediate(
            &self,
            state: &Vector<T, Const<4>, Owned<T, Const<4>>>,
            out: &mut Vector2<T>,
        ) {
            out[0] = state[0];
            out[1] = state[1];
        }
    }

    // MotionModel for evolving the position
    pub struct MotionModel<T: RealField + NumCast + Copy + Default, const S: usize> {
        _marker: PhantomData<T>,
    }

    impl<T: RealField + NumCast + Copy + Default, const S: usize> NonlinearProcessModel<T, 2, S>
        for MotionModel<T, S>
    {
        fn f(&self, state: &mut VectorViewMut<T, Const<2>, Const<1>, Const<{ S }>>, _dt: T) {
            let (vx, vy) = separate_state_vars_2(state);
            *vx = T::one();
            *vy = T::one();
        }

        fn process_noise(
            &self,
            process_noise: &mut MatrixViewMut<T, Const<2>, Const<2>, Const<1>, Const<S>>,
            _dt: T,
        ) {
            process_noise[(0, 0)] = T::one();
            // todo!()
            // Placeholder dynamics for the example
        }

        fn transition_jacobian(
            &self,
            _state: &VectorViewMut<T, Const<2>, Const<1>, Const<S>>,
            jacobian: &mut MatrixViewMut<T, Const<2>, Const<2>, Const<1>, Const<S>>,
            _dt: T,
        ) {
            jacobian[(1, 1)] = T::one();
            // todo!()
            // Placeholder dynamics for the example
        }
    }

    struct TemperatureModel<T: RealField + NumCast + Copy + Default, const S: usize> {
        _marker: PhantomData<T>,
    }

    struct MotionModelMapping;
    impl IntermediateStateStateMapping<f64, 2, 4> for MotionModelMapping {
        type Start = Const<0>;
        type End = Const<1>;
    }

    struct TempuratureModelMapping;
    impl IntermediateStateStateMapping<f64, 1, 4> for TempuratureModelMapping {
        type Start = Const<2>;
        type End = Const<2>;
    }

    impl<T: RealField + NumCast + Copy + Default, const S: usize> NonlinearProcessModel<T, 1, S>
        for crate::struct_based::extended_kalman_filter::tests::TemperatureModel<T, S>
    {
        fn f(&self, state: &mut VectorViewMut<T, Const<1>, Const<1>, Const<{ S }>>, _dt: T) {
            let (temp,) = separate_state_vars_1(state);
            *temp = T::one();
        }

        fn process_noise(
            &self,
            process_noise: &mut MatrixViewMut<T, Const<1>, Const<1>, Const<1>, Const<S>>,
            _dt: T,
        ) {
            process_noise[(0, 0)] = T::one();
            // todo!()
            // Placeholder dynamics for the example
        }

        fn transition_jacobian(
            &self,
            _state: &VectorViewMut<T, Const<1>, Const<1>, Const<S>>,
            jacobian: &mut MatrixViewMut<T, Const<1>, Const<1>, Const<1>, Const<S>>,
            _dt: T,
        ) {
            jacobian[(0, 0)] = T::one();
            // todo!()
            // Placeholder dynamics for the example
        }
    }

    #[test]
    fn test_efk_simple() {
        let mut ekf = ExtendedKalmanFilter {
            x: Vector4::new(0.0, 0.0, 1.0, 1.0),
            p: Matrix4::identity(),
        };

        let gps_measurement = Vector2::new(1.0, 1.0);

        let gps_model = GpsMeasurementModel;
        let gps_mapping = GpsIntermediateMeasurmentStateMapping;
        let mut gps_workspace = GenericNonlinearUpdateWorkspace::new(Default::default());
        ekf.update(
            &gps_measurement,
            &gps_model,
            &gps_mapping,
            &mut gps_workspace,
        );

        let motion_model = MotionModel::<f64, 4> {
            _marker: Default::default(),
        };
        let motion_transition = MotionModelMapping;
        let mut predict_workspace = GenericNonlinearPredictWorkspace::<f64, 4>::new();
        ekf.predict(
            &motion_model,
            &motion_transition,
            &mut predict_workspace,
            0.0,
        );

        println!("{:?}", ekf.x);
    }
}
