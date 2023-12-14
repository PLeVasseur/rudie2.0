use nalgebra::{ArrayStorage, Const, RawStorageMut, RealField, Vector};
use num_traits::NumCast;

use crate::base::types::KalmanState;
use crate::base::types::IntermediateStateStateMapping;
use crate::base::types::NonlinearPredictWorkspace;
use crate::base::types::NonlinearProcessWithControlModel;
use crate::base::types::NonlinearProcessModel;

// NonlinearPredictChain and NonlinearPredictWithControlChain is basically what was used to help write the proc macro
// to allow us to get any custom number of predictions with or without control models
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
            PM1: NonlinearProcessModel<T, I1, S>,
            PM2: NonlinearProcessModel<T, I2, S>,
            ST1: IntermediateStateStateMapping<T, I1, S>,
            ST2: IntermediateStateStateMapping<T, I2, S>,
            W: NonlinearPredictWorkspace<T, S>,
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
            ST1: IntermediateStateStateMapping<T, I1, S>,
            ST2: IntermediateStateStateMapping<T, I2, S>,
            W: NonlinearPredictWorkspace<T, S>,
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

// This conditionally includes the std library when tests are being run.
#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {

}