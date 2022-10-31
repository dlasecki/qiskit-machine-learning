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
            hidden_units: List[int],
            param_hamiltonian: OperatorBase,
            num_inputs: int,
            num_weights: int,
            sparse: bool,
            sampling: bool,
            output_shape: Union[int, Tuple[int, ...]],
            input_gradients: bool = False,
            init_params: Optional[Dict[Parameter, complex]] = None,
            backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        self._gibbs_state_builder = gibbs_state_builder
        self._temperature = temperature
        self._hidden_units = hidden_units
        self._param_hamiltonian = (
            param_hamiltonian  # TODO is it necessary? gibbs state builder -> qite has a hamiltonian
        )
        self._num_inputs = num_inputs
        self._num_weights = num_weights
        self._sparse = sparse
        self._sampling = sampling
        self._output_shape = output_shape
        self._input_gradients = input_gradients
        self._params = init_params
        self._backend = backend
        self._gibbs_state_sampler = None

        super().__init__(num_inputs, num_weights, sparse, sampling, output_shape, input_gradients)

    @property
    def parameter_values(self):
        """
        Get parameter values from the generator

        Raises:
            NotImplementedError: not implemented
        """
        return self._params

    # replaced by probabilities() from SamplingNN
    # def calc_p_v_qbm(self) -> np.ndarray:
    #     """Calculates a probability sample from a provided Gibbs state sampler including hidden
    #     units."""
    #     sample_with_hidden_units = self._gibbs_state_sampler.sample(self._backend)
    #     p_v_qbm = self._remove_hidden_units(sample_with_hidden_units)
    #     return p_v_qbm

    # already handled in SamplingNN using probabilities()
    # # TODO move to child class, qbm_generative_nn
    # def _forward(
    #         self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    # ) -> Union[np.ndarray, SparseArray]:
    #     # generative does not use input_data
    #     hamiltonian_param_dict = dict(zip(self._param_hamiltonian.ordered_parameters, weights))
    #     self._gibbs_state_sampler = self._gibbs_state_builder.build(
    #         self._param_hamiltonian, self._temperature, hamiltonian_param_dict
    #     )
    #     p_v_qbm = self._forward_generative(weights)
    #
    #     return p_v_qbm

    # already handled in SamplingNN using probability_gradients()
    # def _backward(
    #         self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    # ) -> Dict[Parameter, Union[complex, float]]:
    #
    #     # TODO deal with input_data
    #     hamiltonian_param_dict = dict(zip(self._param_hamiltonian.ordered_parameters, weights))
    #     self._gibbs_state_sampler = self._gibbs_state_builder.build(
    #         self._param_hamiltonian, self._temperature, hamiltonian_param_dict
    #     )
    #     gibbs_ham_gradients = self._gibbs_state_sampler.calc_hamiltonian_gradients(
    #         self._backend, self._gradient_method
    #     )
    #     # TODO return input_gradients, weight_gradients
    #     return gibbs_ham_gradients

    def _sample(
            self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """Returns samples from the network."""
        raise NotImplementedError

    def _probabilities(
            self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        """Returns the sample probabilities."""
        hamiltonian_param_dict = dict(zip(self._param_hamiltonian.ordered_parameters, weights))
        total_num_qubits = self._param_hamiltonian.num_qubits
        self._gibbs_state_sampler = self._gibbs_state_builder.build(
            self._param_hamiltonian, self._temperature, hamiltonian_param_dict
        )
        sample_with_hidden_units = self._gibbs_state_sampler.sample(self._backend)
        p_v_qbm = self._remove_hidden_units(sample_with_hidden_units, total_num_qubits)
        return p_v_qbm

    def _remove_hidden_units(self, sample_with_hidden_units: np.ndarray, total_num_qubits: int) -> np.ndarray:
        """Removes hidden units from a provided probability sample and returns an adjusted
        probability sample over visible units.
        Args:
            sample_with_hidden_units: An array of probabilities sampled from a Gibbs state sampler
                                        that includes hidden units and their measurement
                                        outcomes.
            total_num_qubits: Total number of qubits in the QBM.
        Returns:
            An array of probability samples from visible units only (excluding hidden units).
        """
        kept_num_qubits = total_num_qubits - len(self._hidden_units)

        visible_units_probs = np.zeros(pow(2, kept_num_qubits))
        all_qubit_labels_ints = range(total_num_qubits)

        for qubit_label_int, prob in zip(all_qubit_labels_ints, sample_with_hidden_units):
            reduced_label = self._gibbs_state_sampler._reduce_label(qubit_label_int) # TODO code duplication vs. this somewhat dangerous call
            visible_units_probs[reduced_label] += prob

        return visible_units_probs

    # def _reduce_label(self, label: int) -> int:
    #     """Accepts an integer label that represents a measurement outcome and discards hidden units
    #     registers in the label.
    #     Args:
    #         label: An integer label that represents a measurement outcome.
    #     Returns:
    #         A reduced label after discarding indices of hidden units.
    #     """
    #     cnt = len(bin(label)) - 2
    #     cnt2 = 0
    #     reduced_label_bits = []
    #     while cnt:
    #         bit = label & 1
    #         label = label >> 1
    #         if cnt2 not in self._hidden_units:
    #             reduced_label_bits.append(bit)
    #         cnt -= 1
    #         cnt2 += 1
    #     reduced_label = 0
    #     for bit in reduced_label_bits[::-1]:
    #         reduced_label = (reduced_label << 1) | bit
    #     return reduced_label

    def _probability_gradients(
            self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Union[np.ndarray, SparseArray], Union[np.ndarray, SparseArray]]:
        """Returns the probability gradients."""
        hamiltonian_param_dict = dict(zip(self._param_hamiltonian.ordered_parameters,
        weights))
        self._gibbs_state_sampler = self._gibbs_state_builder.build(
            self._param_hamiltonian, self._temperature, hamiltonian_param_dict
        )
        gibbs_ham_gradients = self._gibbs_state_sampler.calc_hamiltonian_gradients(
            self._backend, self._gradient_method
        )
        # TODO return input_gradients, weight_gradients
        return gibbs_ham_gradients


