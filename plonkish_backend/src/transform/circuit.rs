use halo2_curves::bn256::Fr;
use halo2_proofs::plonk::ConstraintSystem;
use zkwasm_halo2::{
    arithmetic::MultiMillerLoop,
    plonk::{
        convert_constraint_system_fr, Circuit as ZkCircuit, ConstraintSystem as ZkConstraintSystem,
    },
};

use crate::backend::PlonkishCircuitInfo;

#[derive(Debug)]
pub struct ZKWASMCircuit<E: MultiMillerLoop, C: ZkCircuit<E::Scalar>> {
    pub circuit: C,
    pub k: u32,
    pub instances: Vec<E::Scalar>,
}
pub fn get_plonkish_info<E: MultiMillerLoop, T>(
    k: u32,
    circuit: &[T],
    instances: Vec<E::Scalar>,
) -> PlonkishCircuitInfo<E::Scalar>
where
    T: ZkCircuit<E::Scalar>,
{
    let circuit = &circuit[0];
    let mut cs = ZkConstraintSystem::default();
    let config = T::configure(&mut cs);

    let cs: ConstraintSystem<Fr> = convert_constraint_system_fr::<E>(cs);

    let constants = cs.constants().clone();

    // Convert Gate Constraints.
    PlonkishCircuitInfo {
        k: k as usize,
        num_instances: vec![instances.len()],
        preprocess_polys: todo!(),
        num_witness_polys: todo!(),
        num_challenges: todo!(),
        constraints: todo!(),
        lookups: todo!(),
        permutations: todo!(),
        max_degree: Some(cs.degree::<true>()),
    }
}
