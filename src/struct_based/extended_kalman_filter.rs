use nalgebra::{Matrix, Matrix2, Matrix4, Vector, Vector2, RealField, Vector4, Vector1, DefaultAllocator, DimName, U1, Dim, U2, Scalar, OMatrix, Dyn, MatrixViewMut, Vector3, Matrix3, ToTypenum, VectorSliceMut, VectorViewMut, Dynamic, RawStorageMut, RawStorage};
use nalgebra::base::{ArrayStorage, Storage};
use nalgebra::base::dimension::{Const};
use nalgebra::storage::Owned;

use num_traits::cast::NumCast;

use itertools::izip;

use typenum::consts::*;
use typenum::Cmp;
use typenum::Less;
use typenum::type_operators::IsLess;
use typenum::type_operators::IsLessOrEqual;
use core::marker::PhantomData;

use core::marker::Copy;
use core::default::Default;
use core::ops::Deref;

use typenum::{Unsigned, UInt, UTerm};

pub trait KalmanState<T, const S: usize>
    where
        T: RealField + NumCast + Copy + Default
{
    fn state_cov(&mut self) -> (
        &mut Vector<T, Const<S>, Owned<T, Const<S>>>,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>
    );
}

pub trait IntermediateStateSize<T: RealField> {
    type IntermediateState: Sized;
}

pub trait NonlinearProcessModel<const S: usize, T>: IntermediateStateSize<T>
    where
        T: RealField + NumCast + Copy + Default
{
    fn f(&self, state: &mut Self::IntermediateState, dt: T);
    fn process_noise(&self, process_noise: &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>);
    fn transition_jacobian(&self, state: &Vector<T, Const<S>, Owned<T, Const<S>>>, jacobian: &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>, dt: T);
}


pub trait NonlinearPredictWorkspace<const S: usize, T, ISS: IntermediateStateSize<T>>
    where
        T: RealField + NumCast + Copy + Default
{
    fn workspace_temps(&mut self) -> (
        &mut ISS::IntermediateState,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
    );
}

pub trait IntermediateStateStateMapping<const S: usize, T, ISS: IntermediateStateSize<T>>
    where
        T: RealField + NumCast + Copy + Default
{
    fn to_process(&self, state: &Vector<T, Const<S>, Owned<T, Const<S>>>, process_state: &mut ISS::IntermediateState);
    fn from_process(&self, process_state: &ISS::IntermediateState, state: &mut Vector<T, Const<S>, ArrayStorage<T, S, 1>>);

    /// Returns the range of indices in the state vector that the intermediate state maps to.
    fn jacobian_range(&self) -> (usize, usize);

    /// Returns the range of indices in the noise matrix that the intermediate state's process noise maps to.
    fn noise_range(&self) -> (usize, usize);
}

pub trait NonlinearPredict<T, const S: usize, ISS, Workspace>: KalmanState<T, S>
    where
        T: RealField + NumCast + Copy + Default,
        ISS: IntermediateStateSize<T>,
        Workspace: NonlinearPredictWorkspace<S, T, ISS>,
{
    fn predict<PM, ST>(
        &mut self,
        process_model: &PM,
        transition: &ST,
        workspace: &mut Workspace,
        dt: T
    )
        where
            ArrayStorage<T, S, S>: Storage<T, Const<S>, Const<S>>,
            PM: NonlinearProcessModel<S, T, IntermediateState=ISS::IntermediateState>,
            ST: IntermediateStateStateMapping<S, T, PM>
    {
        let (process_state, process_noise, transition_jacobian) = workspace.workspace_temps();
        let (state, cov) = self.state_cov();

        transition.to_process(state, process_state);
        process_model.f(process_state, dt);
        transition.from_process(process_state, state);

        process_model.transition_jacobian(state, transition_jacobian, dt);
        process_model.process_noise(process_noise);
        *cov = *transition_jacobian * *cov * transition_jacobian.transpose() + *process_noise;
    }
}

pub trait NonlinearMeasurementModel<const M: usize, const S: usize, T>: IntermediateStateSize<T>
    where
        T: RealField + NumCast + Copy + Default
{
    fn initialize_state(&self) -> Self::IntermediateState;

    fn h(&self, state: &Self::IntermediateState, out: &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>);
    fn measurement_noise(&self) -> Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>;

    fn measurement_jacobian(&self, state: &Vector<T, Const<S>, Owned<T, Const<S>>>,
                            H: &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>);
}

pub trait NonlinearUpdateWorkspace<const S: usize, const M: usize, T, ISS: IntermediateStateSize<T>>
    where
        T: RealField + NumCast + Copy + Default
{
    fn workspace_temps(&mut self) -> (&mut ISS::IntermediateState,
                                      &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
                                      &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
                                      &mut Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
                                      &mut Matrix<T, Const<S>, Const<M>, ArrayStorage<T, S, M>>,
                                      &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
                                      &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
                                      &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
                                      &mut Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>);
}

pub trait IntermediateMeasurmentStateMapping<const S: usize, const M: usize, T, ISS: IntermediateStateSize<T>>
    where
        T: RealField + NumCast + Copy + Default
{
    fn to_intermediate(&self, state: &Vector<T, Const<S>, Owned<T, Const<S>>>, out: &mut ISS::IntermediateState);
}


pub struct GenericNonlinearUpdateWorkspace<T: RealField + NumCast + Copy + Default, const S: usize, const M: usize, MM: NonlinearMeasurementModel<M, S, T>> {
    intermediate_state: MM::IntermediateState,
    predicted_measurement: Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
    temp: Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
    S_: Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
    K: Matrix<T, Const<S>, Const<M>, ArrayStorage<T, S, M>>,
    H: Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
    y: Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
    I: Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
    S_inv: Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
}

impl<T: RealField + NumCast + Copy + Default, const S: usize, const M: usize, MM: NonlinearMeasurementModel<M, S, T>> NonlinearUpdateWorkspace<S, M, T, MM> for GenericNonlinearUpdateWorkspace<T, S, M, MM> {
    fn workspace_temps(&mut self) -> (&mut MM::IntermediateState,
                                      &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
                                      &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
                                      &mut Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>,
                                      &mut Matrix<T, Const<S>, Const<M>, ArrayStorage<T, S, M>>,
                                      &mut Matrix<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>,
                                      &mut Vector<T, Const<M>, ArrayStorage<T, M, 1>>,
                                      &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
                                      &mut Matrix<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>)
    {
        (&mut self.intermediate_state,
         &mut self.predicted_measurement,
         &mut self.temp,
         &mut self.S_,
         &mut self.K,
         &mut self.H,
         &mut self.y,
         &mut self.I,
         &mut self.S_inv)
    }
}

impl<T: RealField + NumCast + Copy + Default, const S: usize, const M: usize, MM: NonlinearMeasurementModel<M, S, T>> GenericNonlinearUpdateWorkspace<T, S, M, MM> {
    pub fn new(initial_intermediate_state: MM::IntermediateState) -> Self {
        Self {
            intermediate_state: initial_intermediate_state,
            predicted_measurement: Vector::<T, Const<M>, ArrayStorage<T, M, 1>>::zeros(),
            temp: Matrix::<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>::zeros(),
            S_: Matrix::<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>::zeros(),
            K: Matrix::<T, Const<S>, Const<M>, ArrayStorage<T, S, M>>::zeros(),
            H: Matrix::<T, Const<M>, Const<S>, ArrayStorage<T, M, S>>::zeros(),
            y: Vector::<T, Const<M>, ArrayStorage<T, M, 1>>::zeros(),
            I: Matrix::<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>::identity(),
            S_inv: Matrix::<T, Const<M>, Const<M>, ArrayStorage<T, M, M>>::zeros(),
        }
    }
}

pub trait NonlinearUpdate<T, const S: usize, const M: usize, ISS, Workspace>: KalmanState<T, S>
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
        MM: NonlinearMeasurementModel<M, S, T, IntermediateState=ISS::IntermediateState>,
        SM: IntermediateMeasurmentStateMapping<S, M, T, ISS>,
        ArrayStorage<T, M, S>: Storage<T, Const<M>, Const<S>>,
        ArrayStorage<T, S, S>: Storage<T, Const<S>, Const<S>>
    {
        let (intermediate_state, predicted_measurement, temp, S_, K, H, y, I, S_inv) = workspace.workspace_temps();
        let (state, cov) = self.state_cov();
        mapping.to_intermediate(state, intermediate_state);
        measurement_model.h(intermediate_state, predicted_measurement);

        measurement_model.measurement_jacobian(state, H);
        *temp = *H * *cov;

        *S_ = *temp * H.transpose() + measurement_model.measurement_noise();

        // Compute the Kalman gain using the previously computed S_
        *S_inv = S_.try_inverse().expect("Matrix is not invertible!");
        *K = *cov * H.transpose() * *S_inv;

        // Compute the measurement residual
        *y = z - *predicted_measurement;

        // Update the state estimate
        *state = *state + *K * *y;

        // Update the error covariance matrix using workspace matrices
        *cov = (*I - *K * *H) * *cov;
    }
}

pub struct ExtendedKalmanFilter<T: RealField, const S: usize> {
    x: Vector<T, Const<S>, Owned<T, Const<S>>>,
    P: Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
}

impl <T: RealField + NumCast + Copy + Default, const S: usize> KalmanState<T, S> for ExtendedKalmanFilter<T, S> {
    fn state_cov(&mut self) -> (&mut Vector<T, Const<S>, Owned<T, Const<S>>>, &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>) {
        (&mut self.x, &mut self.P)
    }
}

impl <T: RealField + NumCast + Copy + Default, const S: usize, const M: usize, ISS: IntermediateStateSize<T>, Workspace: NonlinearUpdateWorkspace<S, M, T, ISS>> NonlinearUpdate<T, S, M, ISS, Workspace> for ExtendedKalmanFilter<T, S> {}
impl <T: RealField + NumCast + Copy + Default, const S: usize, ISS: IntermediateStateSize<T>, Workspace: NonlinearPredictWorkspace<S, T, ISS>> NonlinearPredict<T, S, ISS, Workspace> for ExtendedKalmanFilter<T, S> {}

// -- Example of two distinct states in the state space --

// Define the constants
pub const STATE_SIZE: usize = 3;
pub const POS_STATE_SIZE: usize = 2;
pub const TEMP_STATE_SIZE: usize = 1;

struct KinematicsMeasurementModel;

impl<T: RealField + NumCast + Copy + Default> IntermediateStateSize<T> for KinematicsMeasurementModel {
    type IntermediateState = Vector2<T>;
}

impl<T: RealField + NumCast + Copy + Default, const S: usize> NonlinearMeasurementModel<2, S, T> for KinematicsMeasurementModel {

    fn initialize_state(&self) -> Self::IntermediateState {
        Vector2::<T>::zeros()
    }

    fn h(&self, state: &Self::IntermediateState, out: &mut Vector2<T>) {
        out[0] = state[0];
        out[1] = state[1];
    }

    fn measurement_noise(&self) -> Matrix2<T> {
        Matrix2::new(
            NumCast::from(0.01).unwrap_or_else(T::zero),
            NumCast::from(0.0).unwrap_or_else(T::zero),
            NumCast::from(0.0).unwrap_or_else(T::zero),
            NumCast::from(0.01).unwrap_or_else(T::zero)
        )
    }

    fn measurement_jacobian(&self, _state: &Vector<T, Const<S>, Owned<T, Const<S>>>, H: &mut Matrix<T, Const<2>, Const<S>, ArrayStorage<T, 2, S>>) {
        // Fill in the Jacobian matrix `H` according to your measurement model's relation with the state
        // Given the example, assuming that only the first two state variables relate to the measurement model:
        H.fill(T::zero());
        H[(0, 0)] = T::one();
        H[(1, 1)] = T::one();
    }
}

struct KinematicsIntermediateMeasurmentStateMapping;

impl<T: RealField + NumCast + Copy + Default> IntermediateMeasurmentStateMapping<4, 2, T, KinematicsMeasurementModel> for KinematicsIntermediateMeasurmentStateMapping {
    // type Workspace = GenericNonlinearUpdateWorkspace<T, 4, 2, KinematicsMeasurementModel>;

    fn to_intermediate(&self, state: &Vector4<T>, out: &mut Vector2<T>) {
        out[0] = state[2];
        out[1] = state[3];
    }
}

struct GpsMeasurementModel;

impl<T: RealField + NumCast + Copy + Default> IntermediateStateSize<T> for GpsMeasurementModel {
    type IntermediateState = Vector2<T>;
}

impl<T: RealField + NumCast + Copy + Default, const S: usize> NonlinearMeasurementModel<2, S, T> for GpsMeasurementModel {

    fn initialize_state(&self) -> Self::IntermediateState { // Implement the initialization here
        Vector2::<T>::zeros()
    }

    fn h(&self, state: &Self::IntermediateState, out: &mut Vector<T, Const<2>, ArrayStorage<T, 2, 1>>) {
        out[0] = state[0];
        out[1] = state[1];
    }

    fn measurement_noise(&self) -> Matrix<T, Const<2>, Const<2>, ArrayStorage<T, 2, 2>> {
        Matrix2::new(
            NumCast::from(0.01).unwrap_or_else(T::zero),
            NumCast::from(0.0).unwrap_or_else(T::zero),
            NumCast::from(0.0).unwrap_or_else(T::zero),
            NumCast::from(0.01).unwrap_or_else(T::zero)
        )
    }

    fn measurement_jacobian(&self, _state: &Vector<T, Const<S>, Owned<T, Const<S>>>, H: &mut Matrix<T, Const<2>, Const<S>, ArrayStorage<T, 2, S>>) {
        // Fill in the Jacobian matrix `H` according to your measurement model's relation with the state
        // Given the example, assuming that only the first two state variables relate to the measurement model:
        H.fill(T::zero());
        H[(0, 0)] = T::one();
        H[(1, 1)] = T::one();
    }
}

struct GpsIntermediateMeasurmentStateMapping;

impl<T: RealField + NumCast + Copy + Default> IntermediateMeasurmentStateMapping<4, 2, T, GpsMeasurementModel> for GpsIntermediateMeasurmentStateMapping {

    fn to_intermediate(&self, state: &Vector<T, Const<4>, Owned<T, Const<4>>>, out: &mut Vector2<T>) {
        out[0] = state[0];
        out[1] = state[1];
    }
}

// Define the robot state
#[derive(Clone, Debug)]
pub struct RobotState<T: RealField> {
    position: Vector2<T>,
    temperature: T,
}

// MotionModel for evolving the position
pub struct MotionModel<T: RealField> {
    velocity: Vector2<T>,
}

impl<T: RealField> IntermediateStateSize<T> for MotionModel<T> {
    type IntermediateState = Vector2<T>;
}

impl<T: RealField + NumCast + Copy + Default> NonlinearProcessModel<POS_STATE_SIZE, T> for MotionModel<T> {

    fn f(&self, state: &mut Self::IntermediateState, dt: T) {
        *state += self.velocity * dt;
    }

    fn process_noise(&self, process_noise: &mut Matrix2<T>) {
        // Here we just add some simple noise for the demonstration
        process_noise.fill_diagonal(T::from_f64(0.1).unwrap());
    }

    fn transition_jacobian(&self, _: &Vector2<T>, jacobian: &mut Matrix2<T>, _: T) {
        jacobian.fill_diagonal(T::one());
    }
}

// TemperatureModel for evolving the temperature
pub struct TemperatureModel;

impl<T: RealField> IntermediateStateSize<T> for TemperatureModel {
    type IntermediateState = Vector1<T>;
}

impl<T: RealField + NumCast + Copy + Default> NonlinearProcessModel<TEMP_STATE_SIZE, T> for TemperatureModel {

    fn f(&self, _state: &mut Self::IntermediateState, _: T) {
        // In this example, the temperature doesn't change, but in a more complex model,
        // we would have equations to evolve the temperature based on some conditions.
    }

    fn process_noise(&self, process_noise: &mut Matrix<T, Const<TEMP_STATE_SIZE>, Const<TEMP_STATE_SIZE>, ArrayStorage<T, TEMP_STATE_SIZE, TEMP_STATE_SIZE>>) {
        // Add some temperature measurement noise
        process_noise[(0, 0)] = T::from_f64(0.5).unwrap();
    }

    fn transition_jacobian(&self, _: &Vector1<T>, jacobian: &mut Matrix<T, Const<TEMP_STATE_SIZE>, Const<TEMP_STATE_SIZE>, ArrayStorage<T, TEMP_STATE_SIZE, TEMP_STATE_SIZE>>, _: T) {
        jacobian[(0, 0)] = T::one();
    }
}

// This conditionally includes the std library when tests are being run.
#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use std::{println, vec};
    use nalgebra::{Matrix6, Vector6};

    #[test]
    fn test_efk_simple() {
        let mut ekf = ExtendedKalmanFilter {
            x: Vector4::new(0.0, 0.0, 1.0, 1.0),
            P: Matrix4::identity(),
        };

        let gps_measurement = Vector2::new(1.0, 1.0);
        let kinematics_measurement = Vector2::new(0.9, 1.1);

        let gps_model = GpsMeasurementModel;
        let gps_mapping = GpsIntermediateMeasurmentStateMapping;
        let mut gps_workspace = GenericNonlinearUpdateWorkspace::new(Default::default());
        ekf.update(&gps_measurement, &gps_model, &gps_mapping, &mut gps_workspace);

        let kinematics_model = KinematicsMeasurementModel;
        let kinematics_mapping = KinematicsIntermediateMeasurmentStateMapping;
        let mut kinematics_workspace = GenericNonlinearUpdateWorkspace::new(Default::default());
        ekf.update(&kinematics_measurement, &kinematics_model, &kinematics_mapping, &mut kinematics_workspace);

        println!("{:?}", ekf.x);
    }
}
