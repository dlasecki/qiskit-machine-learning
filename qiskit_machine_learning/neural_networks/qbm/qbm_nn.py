# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from abc import ABC
from typing import Union, List, Optional, Tuple, Dict

import numpy as np
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.algorithms.objective_functions import SparseArray
from qiskit_machine_learning.neural_networks import SamplingNeuralNetwork


class QbmNN(ABC, SamplingNeuralNetwork):
    def __init__(
            self,
            gibbs_state_builder: GibbsStateBuilder,
            temperature: float,  # TODO hide this as a class field in GibbsStateBuilder?
            visible_units: List[int],
            param_hamiltonian: OperatorBase,
            num_inputs: int,
            num_weights: int,
            sparse: bool,
            output_shape: Union[int, Tuple[int, ...]],
            input_gradients: bool = False,
            init_params: Optional[Dict[Parameter, complex]] = None,
            backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        self._gibbs_state_builder = gibbs_state_builder
        self._temperature = temperature
        self._visible_units = visible_units
        self._param_hamiltonian = (
            param_hamiltonian  # TODO is it necessary? gibbs state builder -> qite has a hamiltonian
        )
        self._num_inputs = num_inputs
        self._num_weights = num_weights
        self._sparse = sparse
        self._output_shape = output_shape
        self._input_gradients = input_gradients
        self._params = init_params
        self._backend = backend
        self._gibbs_state_sampler = None

    @property
    def parameter_values(self):
        """
        Get parameter values from the generator

        Raises:
            NotImplementedError: not implemented
        """
        return self._params

    def _forward_generative(self, weights: Optional[np.ndarray]) -> Union[np.ndarray, SparseArray]:

        # TODO remove this method?
        return self.calc_p_v_qbm()

    def calc_p_v_qbm(self) -> np.ndarray:
        """Calculates a probability sample from a provided Gibbs state sampler including hidden
        units."""
        sample_with_hidden_units = self._gibbs_state_sampler.sample(self._backend)
        p_v_qbm = self._remove_hidden_units(sample_with_hidden_units)
        return p_v_qbm

    def _remove_hidden_units(self, sample_with_hidden_units: np.ndarray) -> np.ndarray:
        """Removes hidden units from a provided probability sample and returns an adjusted
        probability sample over visible units."""
        pass

    # TODO move to child class, qbm_generative_nn
    def _forward(
            self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        # generative does not use input_data
        hamiltonian_param_dict = dict(zip(self._param_hamiltonian.ordered_parameters, weights))
        self._gibbs_state_sampler = self._gibbs_state_builder.build(
            self._param_hamiltonian, self._temperature, hamiltonian_param_dict
        )
        p_v_qbm = self._forward_generative(weights)

        return p_v_qbm

    def _backward(
            self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Dict[Parameter, Union[complex, float]]:

        # TODO deal with input_data
        hamiltonian_param_dict = dict(zip(self._param_hamiltonian.ordered_parameters, weights))
        self._gibbs_state_sampler = self._gibbs_state_builder.build(
            self._param_hamiltonian, self._temperature, hamiltonian_param_dict
        )
        gibbs_ham_gradients = self._gibbs_state_sampler.calc_hamiltonian_gradients(
            self._backend, self._gradient_method
        )

        return gibbs_ham_gradients

    def _calc_p_v_from_data(self) -> np.ndarray:
        pass

    # TODO move to child classes, will use existing LossFunction impl
    def calc_obj_fun_grad(
            self,
            p_v_data: np.ndarray,
            measurement_op: OperatorBase,
            temperature: float,
            hamiltonian_param_dict: Dict[Parameter, float],
            gradient_method: str = "param_shift",
    ) -> Dict[Parameter, float]:
        gradient_params = self._param_hamiltonian.ordered_parameters  # validate with Gibbs state

        gibbs_state = self._gibbs_state_builder.build(self._param_hamiltonian, temperature)
        p_v_qbm = self._calc_p_v_qbm(
            gibbs_state, hamiltonian_param_dict
        )  # H params should be bound already
        gibbs_ham_gradients = gibbs_state.calc_hamiltonian_gradients(
            gradient_params, measurement_op, gradient_method
        )  # H params should be bound already

        obj_fun_grads = {}
        # TODO having them as an ordered np.ndarray would allow using sum which could be faster
        for hamiltonian_param in gibbs_ham_gradients.keys():
            gradient = -p_v_data / p_v_qbm * gibbs_ham_gradients[hamiltonian_param]
            obj_fun_grads[hamiltonian_param] = gradient

        return obj_fun_grads
