use rudie_proc_macro::{generate_less_than_impls};

use nalgebra::base::dimension::{Const};
use nalgebra::{ArrayStorage, DimName, Matrix, MatrixViewMut, Owned, RawStorage, RawStorageMut, RealField, Storage, Vector, VectorViewMut};
use num_traits::NumCast;

pub trait LessThan<RHS> {}

generate_less_than_impls!();

pub trait DimNameToUsize {
    const VALUE: usize;
}

impl<const N: usize> DimNameToUsize for Const<N> {
    const VALUE: usize = N;
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

pub trait NonlinearProcessModel3<T, const I: usize, const S: usize>
    where
        T: RealField + NumCast + Copy + Default,
        ArrayStorage<T, I, I>: Storage<T, Const<I>, Const<I>>
{
    fn f(&self, state: &mut VectorViewMut<T, Const<I>, Const<1>, Const<{ S }>>, dt: T);
    fn process_noise(&self, process_noise: &mut MatrixViewMut<T, Const<I>, Const<I>, Const<1>, Const<S>>, dt: T);
    fn transition_jacobian(&self, state: &VectorViewMut<T, Const<I>, Const<1>, Const<S>>, jacobian: &mut MatrixViewMut<T, Const<I>, Const<I>, Const<1>, Const<S>>, dt: T);
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

pub trait IntermediateStateStateMapping3<T, const I: usize, const S: usize>
    where
        T: RealField + NumCast + Copy + Default,
{
    type Start: DimName + DimNameToUsize + LessThan<Self::End>;
    type End: DimName + DimNameToUsize + LessThan<Const<S>>;

    const START: usize = <Self::Start as DimNameToUsize>::VALUE;

    fn to_process<'a>(&'a self, state: &'a mut Vector<T, Const<S>, ArrayStorage<T, S, 1>>) -> VectorViewMut<'a, T, Const<I>, Const<1>, Const<S>>
        where
            ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            ArrayStorage<T, S, 1>: RawStorage<T, Const<S>, RStride = Const<1>, CStride = Const<S>>
    {
        state.fixed_view_mut::<I, 1>(Self::START, 0)
    }

    fn jacobian_matrix<'a>(&'a self, full_jacobian: &'a mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>) -> MatrixViewMut<'a, T, Const<I>, Const<I>, Const<1>, Const<S>>
        where
            ArrayStorage<T, S, S>: RawStorageMut<T, Const<S>, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            ArrayStorage<T, S, S>: RawStorage<T, Const<S>, Const<S>, RStride = Const<1>, CStride = Const<S>>
    {
        full_jacobian.fixed_view_mut::<I, I>(Self::START, Self::START)
    }

    fn noise_matrix<'a>(&'a self, full_noise: &'a mut Matrix<T, Const<S>, Const<S>, ArrayStorage<T, S, S>>) -> MatrixViewMut<'a, T, Const<I>, Const<I>, Const<1>, Const<S>>
        where
            ArrayStorage<T, S, S>: RawStorageMut<T, Const<S>, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            ArrayStorage<T, S, S>: RawStorage<T, Const<S>, Const<S>, RStride = Const<1>, CStride = Const<S>>
    {
        full_noise.fixed_view_mut::<I, I>(Self::START, Self::START)
    }
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
