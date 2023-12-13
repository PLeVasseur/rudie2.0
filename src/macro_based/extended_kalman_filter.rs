use nalgebra::{Const, Matrix, RawStorageMut, RealField};
use nalgebra::base::{ArrayStorage};

use num_traits::cast::NumCast;

use rudie_proc_macro::generate_nonlinear_predict_chain;
use rudie_proc_macro::generate_all_nonlinear_predict_chain;
use rudie_proc_macro::generate_separate_state_vars;
use rudie_proc_macro::generate_all_separate_state_vars;

use crate::base::types::NonlinearProcessModel3;
use crate::base::types::KalmanState;
use crate::base::types::IntermediateStateStateMapping3;

pub mod proc_macros {
    pub use rudie_proc_macro::generate_nonlinear_predict_chain_custom;
}

generate_all_separate_state_vars!();

generate_all_nonlinear_predict_chain!();

pub trait NonlinearPredictWorkspace3<T, const S: usize>
    where
        T: RealField + NumCast + Copy + Default,
{
    fn workspace_temps(&mut self) -> (
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
    );
}

pub struct GenericNonlinearPredictWorkspace3<T: RealField + NumCast + Copy + Default, const S: usize> {
    process_noise: Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
    transition_jacobian: Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
}

impl<T: RealField + NumCast + Copy + Default, const S: usize> NonlinearPredictWorkspace3<T, S> for GenericNonlinearPredictWorkspace3<T, S> {
    fn workspace_temps(&mut self) -> (
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
    ) {
        (&mut self.process_noise, &mut self.transition_jacobian)
    }
}

impl<T: RealField + NumCast + Copy + Default, const S: usize> GenericNonlinearPredictWorkspace3<T, S> {
    pub fn new() -> Self {
        Self {
            process_noise: Matrix::<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>::zeros(),
            transition_jacobian: Matrix::<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>::zeros(),
        }
    }
}

// This conditionally includes the std library when tests are being run.
#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use super::proc_macros::generate_nonlinear_predict_chain_custom;
    use crate::base::types::NonlinearProcessWithControlModel;
    use core::marker::PhantomData;
    use std::{println};
    use nalgebra::{Matrix6, MatrixViewMut, Vector, Vector6, VectorViewMut};

    generate_nonlinear_predict_chain_custom!(
        MyNonlinearPredictChainRenamedGenerics,
        NonlinearProcessWithControlModel<T, B, C, S>,
        NonlinearProcessModel3<T, F, S>
    );

    generate_nonlinear_predict_chain_custom!(
        MyNonlinearPredictChainWithNoControlInputs,
        NonlinearProcessModel3<T, I1, S>,
        NonlinearProcessModel3<T, I2, S>
    );

    #[derive(Debug)]
    struct SimpleKalmanFilter<T: RealField + NumCast + Copy + Default, const S: usize> {
        state: Vector<T, Const<S>, ArrayStorage<T, S, 1>>,
        covariance: Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>,
    }

    impl<T: RealField + NumCast + Copy + Default, const S: usize> SimpleKalmanFilter<T, S> {
        pub fn new( set_state: Vector<T, Const<S>, ArrayStorage<T, S, 1>>, set_covariance: Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>) -> Self {
            Self { state: set_state, covariance: set_covariance }
        }
    }

    impl<T: RealField + NumCast + Copy + Default, const S: usize> KalmanState<T, S> for SimpleKalmanFilter<T, S> {
        fn state_cov(&mut self) -> (&mut Vector<T, Const<S>, ArrayStorage<T, S, 1>>, &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>) {
            (&mut self.state, &mut self.covariance)
        }
    }

    impl <T: RealField + NumCast + Copy + Default, const S: usize> MyNonlinearPredictChainWithNoControlInputs<T, S> for SimpleKalmanFilter<T, S> {}

    // Define the first NonlinearProcessModel3
    struct Model1<T: RealField + NumCast + Copy + Default, const S: usize> {
        _marker: PhantomData<T>,
    }
    impl <T: RealField + NumCast + Copy + Default, const S: usize> NonlinearProcessModel3<T, 3, S> for Model1<T, S> {

        fn f(&self, state: &mut VectorViewMut<T, Const<3>, Const<1>, Const<{ S }>>, _dt: T) {
            let (one, two, three) = separate_state_vars_3(state);
            *one = T::one();
            *two = T::one();
            *three = T::one();
            // todo!()
            // Placeholder dynamics for the example
        }

        fn process_noise(&self, process_noise: &mut MatrixViewMut<T, Const<3>, Const<3>, Const<1>, Const<S>>, _dt: T) {
            process_noise[(0, 0)] = T::one();
            // todo!()
            // Placeholder dynamics for the example
        }

        fn transition_jacobian(&self, _state: &VectorViewMut<T, Const<3>, Const<1>, Const<S>>, jacobian: &mut MatrixViewMut<T, Const<3>, Const<3>, Const<1>, Const<S>>, _dt: T) {
            jacobian[(1, 1)] = T::one();
            // todo!()
            // Placeholder dynamics for the example
        }
    }

    // Define the first NonlinearProcessModel3
    struct Model2<T: RealField + NumCast + Copy + Default, const S: usize> {
        _marker: PhantomData<T>,
    }
    impl <T: RealField + NumCast + Copy + Default, const S: usize> NonlinearProcessModel3<T, 3, S> for Model2<T, S> {
        fn f(&self, state: &mut VectorViewMut<T, Const<3>, Const<1>, Const<{ S }>>, _dt: T) {
            state[0] = T::one();
            state[1] = T::one();
            state[2] = T::one();
            // todo!()
        }

        fn process_noise(&self, process_noise: &mut MatrixViewMut<T, Const<3>, Const<3>, Const<1>, Const<S>>, _dt: T) {
            process_noise[(0, 0)] = T::one();
            // todo!()
        }

        fn transition_jacobian(&self, _state: &VectorViewMut<T, Const<3>, Const<1>, Const<S>>, jacobian: &mut MatrixViewMut<T, Const<3>, Const<3>, Const<1>, Const<S>>, _dt: T) {
            jacobian[(1, 1)] = T::one();
            // todo!()
        }
    }

    struct Mapping1;
    impl IntermediateStateStateMapping3<f64, 3, 6> for Mapping1 {
        type Start = Const<0>;
        type End = Const<2>;
    }

    struct Mapping2;
    impl IntermediateStateStateMapping3<f64, 3, 6> for Mapping2 {
        type Start = Const<3>;
        type End = Const<5>;
    }

    #[test]
    fn test_chained_predictions3() {
        let mut filter = SimpleKalmanFilter::new(
            Vector6::zeros(),
            Matrix6::identity(),
        );
        let model1 = Model1::<f64, 6> { _marker: PhantomData };
        let model2 = Model2::<f64, 6> { _marker: PhantomData };
        let mapping1 = Mapping1;
        let mapping2 = Mapping2;
        let mut workspace = GenericNonlinearPredictWorkspace3::<f64, 6>::new();

        filter.predict(
            (&model1, &model2),
            (&mapping1, &mapping2),
            &mut workspace,
            0.1
        );

        println!("{:?}", filter);
    }
}