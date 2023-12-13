#![no_std]

use paste::paste;

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

use rudie_proc_macro::{generate_less_than_impls, generate_nonlinear_predict_chain_custom};
use rudie_proc_macro::generate_nonlinear_predict_chain;
use rudie_proc_macro::generate_all_nonlinear_predict_chain;
use rudie_proc_macro::generate_separate_state_vars;
use rudie_proc_macro::generate_all_separate_state_vars;

pub mod struct_based;

pub trait LessThan<RHS> {}

generate_less_than_impls!();

pub trait DimNameToUsize {
    const VALUE: usize;
}

impl<const N: usize> DimNameToUsize for Const<N> {
    const VALUE: usize = N;
}

pub trait IsEqual<A: DimName, B: DimName> {}

pub struct AssertEqual;

impl<N: DimName> IsEqual<N, N> for AssertEqual {}

pub trait IntermediateStateStateMapping3<T, const I: usize, const S: usize>
    where
        T: RealField + NumCast + Copy + Default,
{
    type Start: DimName + DimNameToUsize + LessThan<Self::End>;
    type End: DimName + DimNameToUsize + LessThan<Const<S>>;

    const start: usize = <Self::Start as DimNameToUsize>::VALUE;

    fn to_process<'a>(&'a self, state: &'a mut Vector<T, Const<S>, ArrayStorage<T, S, 1>>) -> VectorViewMut<'a, T, Const<I>, Const<1>, Const<S>>
    where
        ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>,
        ArrayStorage<T, S, 1>: RawStorage<T, Const<S>, RStride = Const<1>, CStride = Const<S>>
    {
        state.fixed_view_mut::<I, 1>(Self::start, 0)
    }

    fn jacobian_matrix<'a>(&'a self, full_jacobian: &'a mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>) -> MatrixViewMut<'a, T, Const<I>, Const<I>, Const<1>, Const<S>>
        where
            ArrayStorage<T, S, S>: RawStorageMut<T, Const<S>, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            ArrayStorage<T, S, S>: RawStorage<T, Const<S>, Const<S>, RStride = Const<1>, CStride = Const<S>>
    {
        full_jacobian.fixed_view_mut::<I, I>(Self::start, Self::start)
    }

    fn noise_matrix<'a>(&'a self, full_noise: &'a mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>) -> MatrixViewMut<'a, T, Const<I>, Const<I>, Const<1>, Const<S>>
        where
            ArrayStorage<T, S, S>: RawStorageMut<T, Const<S>, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            ArrayStorage<T, S, S>: RawStorage<T, Const<S>, Const<S>, RStride = Const<1>, CStride = Const<S>>
    {
        full_noise.fixed_view_mut::<I, I>(Self::start, Self::start)
    }
}

pub trait NonlinearProcessModel3<T, const I: usize, const S: usize>
    where
        T: RealField + NumCast + Copy + Default,
        ArrayStorage<T, I, I>: Storage<T, Const<I>, Const<I>>
{
    fn f(&self, state: &mut VectorViewMut<T, Const<I>, Const<1>, Const<{ S }>>, dt: T);
    fn process_noise(&self, process_noise: &mut MatrixViewMut<T, Const<I>, Const<I>, Const<1>, Const<S>>, dt: T);
    fn transition_jacobian(&self, state: &VectorViewMut<T, Const<I>, Const<1>, Const<S>>, jacobian: &mut MatrixViewMut<T, Const<I>, Const<I>, Const<1>, Const<S>>, dt: T);
}

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

pub trait NonlinearPredictChain<T, const S: usize>: KalmanState<T, S>
    where
        T: RealField + NumCast + Copy + Default,
{
    fn predict<const I1: usize, const I2: usize, PM1, PM2, ST1, ST2, W>(
        &mut self,
        process_models: (&PM1, &PM2),
        transitions: (&ST1, &ST2),
        workspace: &mut W,
        dt: T
    )
        where
            ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            PM1: NonlinearProcessModel3<T, I1, S>,
            PM2: NonlinearProcessModel3<T, I2, S>,
            ST1: IntermediateStateStateMapping3<T, I1, S>,
            ST2: IntermediateStateStateMapping3<T, I2, S>,
            W: NonlinearPredictWorkspace3<T, S>,
    {
        let (state, cov) = self.state_cov();

        // TODO: Probably less error-prone to zero out the combined transition jacobian and combined_process_before returning it from workspace
        let (combined_transition_jacobian, combined_process_noise) = workspace.workspace_temps();

        let (model1, model2) = process_models;
        let (transition1, transition2) = transitions;

        // Processing for the first model, transition, and workspace
        {
            let mut intermediate_state = transition1.to_process(state);
            model1.f(&mut intermediate_state, dt);
            let mut intermediate_jacobian = transition1.jacobian_matrix(combined_transition_jacobian);
            model1.transition_jacobian(&intermediate_state, &mut intermediate_jacobian, dt);
            let mut intermediate_noise = transition1.noise_matrix(combined_process_noise);
            model1.process_noise(&mut intermediate_noise, dt);
        }
        // Processing for the second model, transition, and workspace
        {
            let mut intermediate_state = transition2.to_process(state);
            model2.f(&mut intermediate_state, dt);
            let mut intermediate_jacobian = transition2.jacobian_matrix(combined_transition_jacobian);
            model2.transition_jacobian(&intermediate_state, &mut intermediate_jacobian, dt);
            let mut intermediate_noise = transition2.noise_matrix(combined_process_noise);
            model2.process_noise(&mut intermediate_noise, dt);
        }

        *cov = *combined_transition_jacobian * *cov * (*combined_transition_jacobian).transpose() + *combined_process_noise;
    }
}

pub trait NonlinearProcessWithControlModel<T, const I: usize, const C: usize, const S: usize>
    where
        T: RealField + NumCast + Copy + Default,
        ArrayStorage<T, I, I>: Storage<T, Const<I>, Const<I>>
{
    fn f(&self, control_input: &Vector<T, Const<C>, ArrayStorage<T, C, 1>>, state: &mut VectorViewMut<T, Const<I>, Const<1>, Const<{ S }>>, dt: T);
    fn process_noise(&self, process_noise: &mut MatrixViewMut<T, Const<I>, Const<I>, Const<1>, Const<S>>, dt: T);
    fn transition_jacobian(&self, state: &VectorViewMut<T, Const<I>, Const<1>, Const<S>>, jacobian: &mut MatrixViewMut<T, Const<I>, Const<I>, Const<1>, Const<S>>, dt: T);
}

pub trait NonlinearPredictWithControlChain<T, const S: usize>: KalmanState<T, S>
    where
        T: RealField + NumCast + Copy + Default,
{
    fn predict<const I1: usize, const I2: usize, const C1: usize, const C2: usize, PM1, PM2, ST1, ST2, W>(
        &mut self,
        process_models: (&PM1, &PM2),
        control_inputs: (&Vector<T, Const<C1>, ArrayStorage<T, C1, 1>>, &Vector<T, Const<C2>, ArrayStorage<T, C2, 1>>),
        transitions: (&ST1, &ST2),
        workspace: &mut W,
        dt: T
    )
        where
            ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            PM1: NonlinearProcessWithControlModel<T, I1, C1, S>,
            PM2: NonlinearProcessWithControlModel<T, I2, C2, S>,
            ST1: IntermediateStateStateMapping3<T, I1, S>,
            ST2: IntermediateStateStateMapping3<T, I2, S>,
            W: NonlinearPredictWorkspace3<T, S>,
    {
        let (state, cov) = self.state_cov();

        // TODO: Probably less error-prone to zero out the combined transition jacobian and combined_process_before returning it from workspace
        let (combined_transition_jacobian, combined_process_noise) = workspace.workspace_temps();

        let (model1, model2) = process_models;
        let (transition1, transition2) = transitions;
        let (control_input1, control_input2) = control_inputs;

        // Processing for the first model, transition, and workspace
        {
            let mut intermediate_state = transition1.to_process(state);
            model1.f(control_input1, &mut intermediate_state, dt);
            let mut intermediate_jacobian = transition1.jacobian_matrix(combined_transition_jacobian);
            model1.transition_jacobian(&intermediate_state, &mut intermediate_jacobian, dt);
            let mut intermediate_noise = transition1.noise_matrix(combined_process_noise);
            model1.process_noise(&mut intermediate_noise, dt);
        }
        // Processing for the second model, transition, and workspace
        {
            let mut intermediate_state = transition2.to_process(state);
            model2.f(control_input2, &mut intermediate_state, dt);
            let mut intermediate_jacobian = transition2.jacobian_matrix(combined_transition_jacobian);
            model2.transition_jacobian(&intermediate_state, &mut intermediate_jacobian, dt);
            let mut intermediate_noise = transition2.noise_matrix(combined_process_noise);
            model2.process_noise(&mut intermediate_noise, dt);
        }

        *cov = *combined_transition_jacobian * *cov * (*combined_transition_jacobian).transpose() + *combined_process_noise;
    }
}

generate_all_nonlinear_predict_chain!();
generate_all_separate_state_vars!();

generate_nonlinear_predict_chain_custom!(
    MyNonlinearPredictChain,
    NonlinearProcessWithControlModel<T, I1, C1, S>,
    NonlinearProcessModel3<T, I2, S>
);

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
// impl <T: RealField + NumCast + Copy + Default, const S: usize> NonlinearPredictChain2<T, S> for SimpleKalmanFilter<T, S> {}

// Define the first NonlinearProcessModel3
struct Model1<T: RealField + NumCast + Copy + Default, const S: usize> {
    _marker: PhantomData<T>,
}
impl <T: RealField + NumCast + Copy + Default, const S: usize> NonlinearProcessModel3<T, 3, S> for Model1<T, S> {

    fn f(&self, state: &mut VectorViewMut<T, Const<3>, Const<1>, Const<{ S }>>, dt: T) {
        let (mut one, mut two, mut three) = separate_state_vars_3(state);
        *one = T::one();
        *two = T::one();
        *three = T::one();
        // todo!()
        // Placeholder dynamics for the example
    }

    fn process_noise(&self, process_noise: &mut MatrixViewMut<T, Const<3>, Const<3>, Const<1>, Const<S>>, dt: T) {
        process_noise[(0, 0)] = T::one();
        // todo!()
        // Placeholder dynamics for the example
    }

    fn transition_jacobian(&self, state: &VectorViewMut<T, Const<3>, Const<1>, Const<S>>, jacobian: &mut MatrixViewMut<T, Const<3>, Const<3>, Const<1>, Const<S>>, dt: T) {
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
    fn f(&self, state: &mut VectorViewMut<T, Const<3>, Const<1>, Const<{ S }>>, dt: T) {
        state[0] = T::one();
        state[1] = T::one();
        state[2] = T::one();
        // todo!()
    }

    fn process_noise(&self, process_noise: &mut MatrixViewMut<T, Const<3>, Const<3>, Const<1>, Const<S>>, dt: T) {
        process_noise[(0, 0)] = T::one();
        // todo!()
    }

    fn transition_jacobian(&self, state: &VectorViewMut<T, Const<3>, Const<1>, Const<S>>, jacobian: &mut MatrixViewMut<T, Const<3>, Const<3>, Const<1>, Const<S>>, dt: T) {
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

pub trait IntermediateStateSize<T: RealField> {
    type IntermediateState: Sized;
}

struct ExampleModel;

impl<T: Scalar + RealField + NumCast + Copy + Default> NonlinearProcessModel2<2, T> for ExampleModel {
    type IntermediateState = Vector2<T>;
    type Rows = U2;
    type Cols = U2;
    type Storage = ArrayStorage<T, 2, 2>;
    type IntermediateJacobian = MatrixWrapper2x2<T>;

    fn f(&self, state: &mut Self::IntermediateState, dt: T) {
        // Just an arbitrary example of updating the state over time.
        state.x = state.x + dt;
        state.y = state.y - dt;
    }

    fn process_noise(&self, mut process_noise: MatrixViewMut<T, Dyn, Dyn, Const<1>, Const<2>>) {
        let noise_values = Matrix2::new(
            T::one(), T::zero(),
            T::zero(), T::one()
        );

        // Here, you use the `.copy_from()` method to copy the values from noise_values into the provided slice.
        // process_noise.copy_from(&noise_values);
        process_noise[(0, 0)] = T::zero();
        // todo!()
    }

    fn transition_jacobian(&self, state: &Self::IntermediateState, jacobian: &mut Self::IntermediateJacobian, dt: T) {
        todo!()
    }
}

macro_rules! generate_matrix_wrappers {
    ($($size:expr),*) => {
        $(
            paste! {
                pub struct [<MatrixWrapper $size x $size>]<T: RealField + NumCast + Copy + Default>(OMatrix<T, Const<$size>, Const<$size>>);

                impl<T: RealField + NumCast + Copy + Default> core::ops::Deref for [<MatrixWrapper $size x $size>]<T> {
                    type Target = OMatrix<T, Const<$size>, Const<$size>>;

                    fn deref(&self) -> &Self::Target {
                        &self.0
                    }
                }

                impl<T: RealField + NumCast + Copy + Default> core::ops::DerefMut for [<MatrixWrapper $size x $size>]<T> {
                    fn deref_mut(&mut self) -> &mut Self::Target {
                        &mut self.0
                    }
                }

                impl<T: RealField + NumCast + Copy + Default> Default for [<MatrixWrapper $size x $size>]<T> {
                    fn default() -> Self {
                        Self(OMatrix::<T, Const<$size>, Const<$size>>::zeros())
                    }
                }
            }
        )*
    };
}

generate_matrix_wrappers!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                          21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);  // You can expand this list for other matrix sizes.

pub trait NonlinearProcessModel2<const S: usize, T>
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState;
    type Rows: DimName;
    type Cols: DimName;
    type Storage: Storage<T, Self::Rows, Self::Cols>;

    type IntermediateJacobian: Deref<Target = Matrix<T, Self::Rows, Self::Cols, Self::Storage>>;

    fn f(&self, state: &mut Self::IntermediateState, dt: T);
    fn process_noise(&self, process_noise: MatrixViewMut<T, Dyn, Dyn, Const<1>, Const<{ S }>>);
    // fn process_noise(&self, process_noise: &mut Self::IntermediateJacobian);
    fn transition_jacobian(&self, state: &Self::IntermediateState, jacobian: &mut Self::IntermediateJacobian, dt: T);
}

pub trait NonlinearPredictWorkspace2<const S: usize, T>
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState;
    type Rows: DimName;
    type Cols: DimName;
    type Storage: Storage<T, Self::Rows, Self::Cols>;

    type IntermediateJacobian: Deref<Target = Matrix<T, Self::Rows, Self::Cols, Self::Storage>>;

    fn workspace_temps(&mut self) -> (
        &mut Self::IntermediateState,
        &mut Self::IntermediateJacobian,
        &mut Self::IntermediateJacobian,
    );
}

pub trait IntermediateStateStateMapping2<const S: usize, T>
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState;
    type Rows: DimName;
    type Cols: DimName;
    type Storage: Storage<T, Self::Rows, Self::Cols>;

    type IntermediateJacobian: Deref<Target = Matrix<T, Self::Rows, Self::Cols, Self::Storage>>;

    fn to_process(&self, state: &Vector<T, Const<S>, Owned<T, Const<S>>>, process_state: &mut Self::IntermediateState);
    fn from_process(&self, process_state: &Self::IntermediateState, state: &mut Vector<T, Const<S>, ArrayStorage<T, S, 1>>);

    fn jacobian_range(&self) -> (usize, usize);
    fn noise_range(&self) -> (usize, usize);
}

pub trait NonlinearPredict2<T, const S: usize>: KalmanState<T, S>
    where
        T: RealField + NumCast + Copy + Default,
{
    fn predict<PM, ST, W>(
        &mut self,
        process_models: &[&PM],
        transitions: &[&ST],
        workspaces: &mut [&mut W],
        dt: T
    )
        where
            PM: NonlinearProcessModel2<S, T>,
            ST: IntermediateStateStateMapping2<S, T, IntermediateState = PM::IntermediateState, IntermediateJacobian = PM::IntermediateJacobian>,
            W: NonlinearPredictWorkspace2<S, T, IntermediateState = PM::IntermediateState, IntermediateJacobian = PM::IntermediateJacobian>,
    {
        let (state, cov) = self.state_cov();
        let mut combined_transition_jacobian = Matrix::<T, Const<S>, Const<S>, <DefaultAllocator as nalgebra::allocator::Allocator<T, Const<S>, Const<S>>>::Buffer>::zeros_generic(Const::<S>, Const::<S>);

        for (model, transition, workspace) in izip!(process_models, transitions, workspaces) {
            let (process_state, _, transition_jacobian) = workspace.workspace_temps();
            transition.to_process(state, process_state);
            model.f(process_state, dt);
            transition.from_process(process_state, state);

            model.transition_jacobian(process_state, transition_jacobian, dt);

            let (start, end) = transition.jacobian_range();
            combined_transition_jacobian
                .slice_mut((start, start), (end-start, end-start))
                .copy_from(&*transition_jacobian);

           let mut combined_process_noise = Matrix::<T, Const<S>, Const<S>, <DefaultAllocator as nalgebra::allocator::Allocator<T, Const<S>, Const<S>>>::Buffer>::zeros_generic(Const::<S>, Const::<S>);
            for (model, transition) in izip!(process_models, transitions) {
                let (start, end) = transition.noise_range();
                model.process_noise(combined_process_noise.view_mut((start, start), (end-start, end-start)));
            }
        }
    }
}

pub trait NonlinearPredict3<T, const S: usize>: KalmanState<T, S>
    where
        T: RealField + NumCast + Copy + Default,
{
    fn predict<PM1, PM2, ST1, ST2, W1, W2>(
        &mut self,
        process_models: (&PM1, &PM2),
        transitions: (&ST1, &ST2),
        workspaces: (&mut W1, &mut W2),
        dt: T
    )
        where
            PM1: NonlinearProcessModel2<S, T>,
            PM2: NonlinearProcessModel2<S, T>,
            ST1: IntermediateStateStateMapping2<S, T, IntermediateState = PM1::IntermediateState, IntermediateJacobian = PM1::IntermediateJacobian>,
            ST2: IntermediateStateStateMapping2<S, T, IntermediateState = PM2::IntermediateState, IntermediateJacobian = PM2::IntermediateJacobian>,
            W1: NonlinearPredictWorkspace2<S, T, IntermediateState = PM1::IntermediateState, IntermediateJacobian = PM1::IntermediateJacobian>,
            W2: NonlinearPredictWorkspace2<S, T, IntermediateState = PM2::IntermediateState, IntermediateJacobian = PM2::IntermediateJacobian>,
    {
        let (state, cov) = self.state_cov();
        let mut combined_transition_jacobian = Matrix::<T, Const<S>, Const<S>, <DefaultAllocator as nalgebra::allocator::Allocator<T, Const<S>, Const<S>>>::Buffer>::zeros_generic(Const::<S>, Const::<S>);

        // Unpack the tuples
        let (model1, model2) = process_models;
        let (transition1, transition2) = transitions;
        let (workspace1, workspace2) = workspaces;

        // Processing for the first model, transition, and workspace
        {
            let (process_state, _, transition_jacobian) = workspace1.workspace_temps();
            transition1.to_process(state, process_state);
            model1.f(process_state, dt);
            transition1.from_process(process_state, state);
            model1.transition_jacobian(process_state, transition_jacobian, dt);

            let (start, end) = transition1.jacobian_range();
            combined_transition_jacobian
                .view_mut((start, start), (end-start, end-start))
                .copy_from(&*transition_jacobian);
        }

        // Processing for the second model, transition, and workspace
        {
            let (process_state, _, transition_jacobian) = workspace2.workspace_temps();
            transition2.to_process(state, process_state);
            model2.f(process_state, dt);
            transition2.from_process(process_state, state);
            model2.transition_jacobian(process_state, transition_jacobian, dt);

            let (start, end) = transition2.jacobian_range();
            combined_transition_jacobian
                .view_mut((start, start), (end-start, end-start))
                .copy_from(&*transition_jacobian);
        }

        let mut combined_process_noise = Matrix::<T, Const<S>, Const<S>, <DefaultAllocator as nalgebra::allocator::Allocator<T, Const<S>, Const<S>>>::Buffer>::zeros_generic(Const::<S>, Const::<S>);

        // Noise processing for the first model and transition
        {
            let (start, end) = transition1.noise_range();
            model1.process_noise(combined_process_noise.view_mut((start, start), (end-start, end-start)));
        }

        // Noise processing for the second model and transition
        {
            let (start, end) = transition2.noise_range();
            model2.process_noise(combined_process_noise.view_mut((start, start), (end-start, end-start)));
        }

        *cov = combined_transition_jacobian * *cov * combined_transition_jacobian.transpose() + combined_process_noise;
    }
}

pub trait KalmanState<T, const S: usize>
where
    T: RealField + NumCast + Copy + Default
{
    fn state_cov(&mut self) -> (
        &mut Vector<T, Const<S>, Owned<T, Const<S>>>,
        &mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>
    );
}

pub struct ProcessModel1;

impl<const S: usize, T> NonlinearProcessModel2<S, T> for ProcessModel1
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState = Vector2<T>;
    type Rows = U2;
    type Cols = U2;
    type Storage = Owned<T, U2, U2>;

    type IntermediateJacobian = MatrixWrapper2x2<T>;

    fn f(&self, state: &mut Self::IntermediateState, _dt: T) {
        // Just an example operation
        state.x *= T::from(1.1).unwrap();
    }

    fn process_noise(&self, mut process_noise: MatrixViewMut<T, Dyn, Dyn, Const<1>, Const<S>>) {
        // Modify process noise
        process_noise[(0, 0)] = T::from(0.1).unwrap();
    }

    fn transition_jacobian(&self, _state: &Self::IntermediateState, jacobian: &mut Self::IntermediateJacobian, _dt: T) {
        **jacobian = OMatrix::<T, U2, U2>::identity();
    }
}

// Associated workspace for ProcessModel1

pub struct Workspace1 <T: RealField + NumCast + Copy + Default> {
    intermediate_state: Vector2<T>,
    intermediate_jacobian1: MatrixWrapper2x2<T>,
    intermediate_jacobian2: MatrixWrapper2x2<T>,
}

impl<const S: usize, T> NonlinearPredictWorkspace2<S, T> for Workspace1<T>
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState = Vector2<T>;
    type Rows = U2;
    type Cols = U2;
    type Storage = Owned<T, U2, U2>;

    type IntermediateJacobian = MatrixWrapper2x2<T>;

    fn workspace_temps(&mut self) -> (
        &mut Self::IntermediateState,
        &mut Self::IntermediateJacobian,
        &mut Self::IntermediateJacobian,
    ) {
        (&mut self.intermediate_state, &mut self.intermediate_jacobian1, &mut self.intermediate_jacobian2)
    }
}

// Intermediate state-state mapping for ProcessModel1
impl<T> IntermediateStateStateMapping2<3, T> for ProcessModel1
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState = Vector2<T>;
    type Rows = U2;
    type Cols = U2;
    type Storage = Owned<T, U2, U2>;

    type IntermediateJacobian = MatrixWrapper2x2<T>;

    fn to_process(&self, state: &Vector<T, Const<3>, Owned<T, Const<3>>>, process_state: &mut Self::IntermediateState) {
        process_state[(0, 0)] = state[(0, 0)];
        process_state[(1, 0)] = state[(1, 0)];
    }

    fn from_process(&self, process_state: &Self::IntermediateState, state: &mut Vector<T, Const<3>, ArrayStorage<T, 3, 1>>) {
        state[(0, 0)] = T::zero();
    }

    fn jacobian_range(&self) -> (usize, usize) {
        (0, 2)
    }

    fn noise_range(&self) -> (usize, usize) {
        (0, 2)
    }
}

pub struct ProcessModel2;

impl<const S: usize, T> NonlinearProcessModel2<S, T> for ProcessModel2
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState = Vector1<T>;
    type Rows = U1;
    type Cols = U1;
    type Storage = Owned<T, U1, U1>;

    type IntermediateJacobian = MatrixWrapper1x1<T>;

    fn f(&self, state: &mut Self::IntermediateState, _dt: T) {
        // Just an example operation
        state.x *= T::from(1.1).unwrap();
    }

    fn process_noise(&self, mut process_noise: MatrixViewMut<T, Dyn, Dyn, Const<1>, Const<S>>) {
        // Modify process noise
        process_noise[(0, 0)] = T::from(0.1).unwrap();
    }

    fn transition_jacobian(&self, _state: &Self::IntermediateState, jacobian: &mut Self::IntermediateJacobian, _dt: T) {
        **jacobian = OMatrix::<T, U1, U1>::identity();
    }
}

// Associated workspace for ProcessModel1
pub struct Workspace2<T: RealField + NumCast + Copy + Default> {
    intermediate_state: Vector1<T>,
    intermediate_jacobian1: MatrixWrapper1x1<T>,
    intermediate_jacobian2: MatrixWrapper1x1<T>,
}

impl<const S: usize, T> NonlinearPredictWorkspace2<S, T> for Workspace2<T>
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState = Vector1<T>;
    type Rows = U1;
    type Cols = U1;
    type Storage = Owned<T, U1, U1>;

    type IntermediateJacobian = MatrixWrapper1x1<T>;

    fn workspace_temps(&mut self) -> (
        &mut Self::IntermediateState,
        &mut Self::IntermediateJacobian,
        &mut Self::IntermediateJacobian,
    ) {
        (&mut self.intermediate_state, &mut self.intermediate_jacobian1, &mut self.intermediate_jacobian2)
    }
}

// Intermediate state-state mapping for ProcessModel1
impl<T> IntermediateStateStateMapping2<3, T> for ProcessModel2
    where
        T: RealField + NumCast + Copy + Default,
{
    type IntermediateState = Vector1<T>;
    type Rows = U1;
    type Cols = U1;
    type Storage = Owned<T, U1, U1>;

    type IntermediateJacobian = MatrixWrapper1x1<T>;

    fn to_process(&self, state: &Vector<T, Const<3>, Owned<T, Const<3>>>, process_state: &mut Self::IntermediateState) {
        // process_state[(0, 0)] = state[(0, 0)];
        // process_state[(1, 0)] = state[(1, 0)];
    }

    fn from_process(&self, process_state: &Self::IntermediateState, state: &mut Vector<T, Const<3>, ArrayStorage<T, 3, 1>>) {
        state[(0, 0)] = T::zero();
    }

    fn jacobian_range(&self) -> (usize, usize) {
        (0, 2)
    }

    fn noise_range(&self) -> (usize, usize) {
        (0, 2)
    }
}

pub struct MyPredictor <T: RealField + NumCast + Copy + Default> {
    state: Vector3<T>,
    covariance: Matrix3<T>,
}

impl <T: RealField + NumCast + Copy + Default> KalmanState<T, 3> for MyPredictor<T> {
    fn state_cov(&mut self) -> (
        &mut Vector<T, Const<3>, Owned<T, Const<3>>>,
        &mut Matrix<T, Const<3>, Const<3>, ArrayStorage<T, 3, 3>>,
    ) {
        (&mut self.state, &mut self.covariance)
    }
}

impl <T: RealField + NumCast + Copy + Default> NonlinearPredict3<T, 3> for MyPredictor<T> {}

// You can create a second process model in a similar manner, i.e., `ProcessModel2`, `Workspace2`, and another implementation for `IntermediateStateStateMapping2`.

// This conditionally includes the std library when tests are being run.
#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use std::{println, vec};
    use nalgebra::{Matrix6, Vector6};

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