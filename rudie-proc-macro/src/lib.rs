extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, GenericArgument, LitInt, PathArguments, Type};

#[proc_macro]
pub fn generate_less_than_impls(_input: TokenStream) -> TokenStream {
    let mut generated_code = quote! {};

    for lhs in 1usize..=127 {
        // starting from 1 to allow 0 as a valid rhs
        for rhs in 0..lhs {
            generated_code.extend(quote! {
                impl LessThan<Const<#lhs>> for Const<#rhs> {}
            });
        }
    }

    generated_code.into()
}

#[proc_macro]
pub fn generate_less_than_or_equal_impls(_input: TokenStream) -> TokenStream {
    let mut generated_code = quote! {};

    for lhs in 1usize..=127 {
        for rhs in 0..=lhs {
            // include lhs in the range for rhs
            generated_code.extend(quote! {
                impl LessThanOrEqual<Const<#lhs>> for Const<#rhs> {}
            });
        }
    }

    generated_code.into()
}

#[proc_macro]
pub fn generate_nonlinear_predict_chain(input: TokenStream) -> TokenStream {
    // Parse the input to get the tuple size.
    let tuple_size: usize = parse_macro_input!(input as LitInt).base10_parse().unwrap();

    // Generate the generic constraints.
    let generic_names: Vec<_> = (1..=tuple_size).map(|i| format_ident!("I{}", i)).collect();
    let pm_names: Vec<_> = (1..=tuple_size).map(|i| format_ident!("PM{}", i)).collect();
    let st_names: Vec<_> = (1..=tuple_size).map(|i| format_ident!("ST{}", i)).collect();
    let pm_constraints = (1..=tuple_size).map(|i| {
        let i_name = format_ident!("I{}", i);
        let pm_name = format_ident!("PM{}", i);
        quote! {
            #pm_name: NonlinearProcessModel<T, #i_name, S>
        }
    });
    let st_constraints = (1..=tuple_size).map(|i| {
        let i_name = format_ident!("I{}", i);
        let st_name = format_ident!("ST{}", i);
        quote! {
            #st_name: IntermediateStateStateMapping<T, #i_name, S>
        }
    });

    // Generate model names and transition names
    let model_names = (1..=tuple_size)
        .map(|i| format_ident!("model{}", i))
        .collect::<Vec<_>>();
    let transition_names = (1..=tuple_size)
        .map(|i| format_ident!("transition{}", i))
        .collect::<Vec<_>>();

    let model_code = model_names.iter().zip(transition_names.iter()).map(|(model_name, transition_name)| {
        quote! {
        {
            let mut intermediate_state = #transition_name.to_process(state);
            #model_name.f(&mut intermediate_state, dt);
            let mut intermediate_jacobian = #transition_name.jacobian_matrix(combined_transition_jacobian);
            #model_name.transition_jacobian(&intermediate_state, &mut intermediate_jacobian, dt);
            let mut intermediate_noise = #transition_name.noise_matrix(combined_process_noise);
            #model_name.process_noise(&mut intermediate_noise, dt);
        }
    }
    });

    // Trait name based on tuple size.
    let trait_name = format_ident!("NonlinearPredictChain{}", tuple_size);

    // Generate the generic constraints without trailing commas.
    let pm_constraints: Vec<_> = pm_constraints.collect();
    let st_constraints: Vec<_> = st_constraints.collect();

    // Generate model code.
    let model_code = model_code.collect::<Vec<_>>();

    let expanded = quote! {
    pub trait #trait_name<T, const S: usize>: KalmanState<T, S>
    where
        T: RealField + NumCast + Copy + Default,
    {
        fn predict<#(const #generic_names: usize,)* #(#pm_names,)* #(#st_names,)* W>(
            &mut self,
            process_models: (#(&#pm_names,)*),
            transitions: (#(&#st_names,)*),
            workspace: &mut W,
            dt: T
        )
        where
            ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>,
            #(#pm_constraints,)*
            #(#st_constraints,)*
            W: NonlinearPredictWorkspace<T, S>,
        {
                let (state, cov) = self.state_cov();

                // TODO: Probably less error-prone to zero out the combined transition jacobian and combined_process_noise before returning it from workspace
                let (combined_transition_jacobian, combined_process_noise) = workspace.workspace_temps();

                let (#(#model_names,)*) = process_models;
                let (#(#transition_names,)*) = transitions;

                #(#model_code)*

                *cov = *combined_transition_jacobian * *cov * (*combined_transition_jacobian).transpose() + *combined_process_noise;
            }
        }
    };

    // println!("{}", expanded.to_string());

    expanded.into()
}

#[proc_macro]
pub fn generate_all_nonlinear_predict_chain(_input: TokenStream) -> TokenStream {
    // Generate the repeated calls
    let repeated_calls: Vec<_> = (2..=127)
        // let repeated_calls: Vec<_> = (2..=127)
        .map(|i| {
            let tokens = quote! {
                generate_nonlinear_predict_chain!(#i);
            };
            tokens
        })
        .collect();

    // Convert the Vec<proc_macro2::TokenStream> to a single TokenStream
    let generated_code = quote! {
        #(#repeated_calls)*
    };

    generated_code.into()
}

#[proc_macro]
pub fn generate_separate_state_vars(input: TokenStream) -> TokenStream {
    // Parse the input to get N.
    let n: usize = parse_macro_input!(input as LitInt).base10_parse().unwrap();

    // Create the identifiers for the pointers (ptr1, ptr2, ..., ptrN).
    let ptr_ids: Vec<_> = (1..=n)
        .map(|i| syn::Ident::new(&format!("ptr{}", i), proc_macro2::Span::call_site()))
        .collect();

    // Create the function name `separate_state_vars_N`.
    let function_name = syn::Ident::new(
        &format!("separate_state_vars_{}", n),
        proc_macro2::Span::call_site(),
    );

    // Construct the body of the function.
    let ptr_initializations: Vec<_> = ptr_ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            quote! {
                let #id: *mut T = &mut state[#index];
            }
        })
        .collect();

    let ptr_dereferences: Vec<_> = ptr_ids
        .iter()
        .map(|id| {
            quote! {
                unsafe { &mut *#id }
            }
        })
        .collect();

    let return_tuples: Vec<_> = ptr_ids
        .iter()
        .map(|_id| {
            quote! {
                &'a mut T
            }
        })
        .collect();

    // Create the final expanded code.
    let expanded = quote! {
        pub fn #function_name<'a, T, const S: usize>(state: &'a mut nalgebra::VectorViewMut<T, nalgebra::Const<#n>, nalgebra::Const<1>, Const<S>>) -> (#(#return_tuples,)*)
            where
                T: RealField + NumCast + Copy + Default
        {
            #(#ptr_initializations)*

            (#(#ptr_dereferences,)*)
        }
    };

    // println!("{}", expanded.to_string());

    expanded.into()
}

#[proc_macro]
pub fn generate_all_separate_state_vars(_input: TokenStream) -> TokenStream {
    // Generate a sequence of numbers from 1 to 127.
    let range: Vec<_> = (1..=127).collect();

    // For each number, generate a call to `generate_separate_state_vars!`.
    let expansions: Vec<_> = range
        .iter()
        .map(|&n| {
            quote! {
                generate_separate_state_vars!(#n);
            }
        })
        .collect();

    // Expand all of these into the final token stream.
    let expanded = quote! {
        #(#expansions)*
    };

    expanded.into()
}

#[proc_macro]
pub fn generate_nonlinear_predict_chain_custom(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ChainGenerator);
    let expanded = input.generate();

    // println!("{}", expanded.to_string());

    expanded.into()
}

struct ChainGenerator {
    chain_name: Ident,
    process_models: Punctuated<ProcessModel, syn::Token![,]>,
}

struct ProcessModel {
    generics: Vec<Type>,
    has_control: bool,
}

impl syn::parse::Parse for ChainGenerator {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let chain_name: proc_macro2::Ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let process_models =
            Punctuated::<ProcessModel, syn::Token![,]>::parse_separated_nonempty(input)?;
        Ok(ChainGenerator {
            chain_name,
            process_models,
        })
    }
}

impl syn::parse::Parse for ProcessModel {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let path: syn::Path = input.parse()?;
        let mut generics = vec![];

        if let PathArguments::AngleBracketed(ref args) = path.segments.last().unwrap().arguments {
            for arg in &args.args {
                if let GenericArgument::Type(ty) = arg {
                    generics.push(ty.clone());
                }
            }
        }

        let has_control = path
            .segments
            .iter()
            .any(|segment| segment.ident == "NonlinearProcessWithControlModel");
        Ok(ProcessModel {
            generics,
            has_control,
        })
    }
}

fn extract_types_with_control(
    slice: &[syn::Type],
) -> Option<(&syn::Type, &syn::Type, &syn::Type, &syn::Type)> {
    if slice.len() >= 4 {
        Some((&slice[0], &slice[1], &slice[2], &slice[3]))
    } else {
        None
    }
}

fn extract_types_without_control(
    slice: &[syn::Type],
) -> Option<(&syn::Type, &syn::Type, &syn::Type)> {
    if slice.len() >= 3 {
        Some((&slice[0], &slice[1], &slice[2]))
    } else {
        None
    }
}

impl ChainGenerator {
    fn generate(&self) -> proc_macro2::TokenStream {
        let chain_name = &self.chain_name;

        let mut where_constraints = vec![quote! {
            ArrayStorage<T, S, 1>: RawStorageMut<T, Const<S>, RStride = Const<1>, CStride = Const<S>>
        }];

        where_constraints.push(quote! {W: NonlinearPredictWorkspace<T, S>});

        // Generate the function arguments and the corresponding body processing
        let mut const_generics = vec![];
        let mut process_model_params = vec![];
        let mut mapping_params = vec![];
        let mut control_inputs_generics = vec![];
        let mut control_input_params = vec![];
        let mut process_function_body = vec![];

        let mut process_model_unpack_tuple = vec![];
        let mut mapping_unpack_tuple = vec![];
        let mut control_input_unpack_tuple = vec![];

        for (i, model) in self.process_models.iter().enumerate() {
            let index = i + 1;
            let pm_ident = format_ident!("PM{}", index);
            let st_ident = format_ident!("ST{}", index);

            let pm_unpack_tuple = format_ident!("pm{}", index);
            let map_unpack_tuple = format_ident!("st{}", index);

            process_model_params.push(quote! { &#pm_ident });
            mapping_params.push(quote! { &#st_ident });

            process_model_unpack_tuple.push(quote! {#pm_unpack_tuple});
            mapping_unpack_tuple.push(quote! {#map_unpack_tuple});

            if model.has_control {
                let types = extract_types_with_control(&model.generics)
                    .unwrap_or_else(|| panic!("Expected at least 4 generics for the model!"));

                let (t_type, i_type, c_type, s_type) = types;

                let ctrl_unpack_tuple = format_ident!("ci{}", index);
                control_input_unpack_tuple.push(quote! {#ctrl_unpack_tuple});

                // Push process model and transition to function args
                const_generics.push(quote! { const #i_type: usize });

                where_constraints.push(quote! {
                    #st_ident: IntermediateStateStateMapping<#t_type, #i_type, #s_type>
                });

                where_constraints.push(quote! {
                    #pm_ident: NonlinearProcessWithControlModel<#t_type, #i_type, #c_type, #s_type>
                });

                control_inputs_generics.push(quote! { const #c_type: usize });

                control_input_params.push(
                    quote! { &Vector<#t_type, Const<#c_type>, ArrayStorage<#t_type, #c_type, 1>> },
                );

                process_function_body.push(quote! {
                {
                    let mut intermediate_state = #map_unpack_tuple.to_process(state);
                    #pm_unpack_tuple.f(#ctrl_unpack_tuple, &mut intermediate_state, dt);
                    let mut intermediate_jacobian = #map_unpack_tuple.jacobian_matrix(combined_transition_jacobian);
                    #pm_unpack_tuple.transition_jacobian(&intermediate_state, &mut intermediate_jacobian, dt);
                    let mut intermediate_noise = #map_unpack_tuple.noise_matrix(combined_process_noise);
                    #pm_unpack_tuple.process_noise(&mut intermediate_noise, dt);
                }
                });
            } else {
                let types = extract_types_without_control(&model.generics)
                    .unwrap_or_else(|| panic!("Expected at least 3 generics for the model!"));

                let (t_type, i_type, s_type) = types;

                const_generics.push(quote! { const #i_type: usize });

                where_constraints.push(quote! {
                    #st_ident: IntermediateStateStateMapping<#t_type, #i_type, #s_type>
                });

                where_constraints.push(quote! {
                    #pm_ident: NonlinearProcessModel<#t_type, #i_type, #s_type>
                });

                process_function_body.push(quote! {
                {
                    let mut intermediate_state = #map_unpack_tuple.to_process(state);
                    #pm_unpack_tuple.f(&mut intermediate_state, dt);
                    let mut intermediate_jacobian = #map_unpack_tuple.jacobian_matrix(combined_transition_jacobian);
                    #pm_unpack_tuple.transition_jacobian(&intermediate_state, &mut intermediate_jacobian, dt);
                    let mut intermediate_noise = #map_unpack_tuple.noise_matrix(combined_process_noise);
                    #pm_unpack_tuple.process_noise(&mut intermediate_noise, dt);
                }
                });
            }
        }

        let requires_control_input = self.process_models.iter().any(|model| model.has_control);

        let control_inputs = if requires_control_input {
            Some(quote! { control_inputs: (#(#control_input_params,)*), })
        } else {
            None
        };

        // Dynamically generate the type arguments for `predict` based on the number of models
        let type_args = self
            .process_models
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let index = syn::Index::from(i + 1);
                let pm_ident = format_ident!("PM{}", index);
                let st_ident = format_ident!("ST{}", index);
                quote! { #pm_ident, #st_ident }
            })
            .collect::<Vec<_>>();

        // Generate the function signature and body
        let function_signature = quote! {
            fn predict<
                #(#const_generics,)*
                #(#control_inputs_generics,)*
                #(#type_args,)*
                W
            >(
                &mut self,
                process_models: (#(#process_model_params,)*),
                #control_inputs
                transitions: (#(#mapping_params,)*),
                workspace: &mut W,
                dt: T
            ) where #(#where_constraints,)*
        };

        let control_inputs_unpacking = if requires_control_input {
            Some(quote! { let (#(#control_input_unpack_tuple,)*) = control_inputs; })
        } else {
            None
        };

        let function_body = quote! {
            {
                let (state, cov) = self.state_cov();
                let (combined_transition_jacobian, combined_process_noise) = workspace.workspace_temps();

                let (#(#process_model_unpack_tuple,)*) = process_models;
                let (#(#mapping_unpack_tuple,)*) = transitions;
                #control_inputs_unpacking

                #(#process_function_body)*

                *cov = *combined_transition_jacobian * *cov * (*combined_transition_jacobian).transpose() + *combined_process_noise;
            }
        };

        quote! {
            pub trait #chain_name<T, const S: usize>: KalmanState<T, S>
            where
                T: RealField + NumCast + Copy + Default,
            {
                #function_signature
                #function_body
            }
        }
    }
}
