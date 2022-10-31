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

"""Test QBM QNN."""

import unittest

from qiskit.opflow import Zero, SummedOp, Z, I

from test import QiskitMachineLearningTestCase, requires_extra_library

from ddt import ddt, data, unpack

import numpy as np

from qiskit import Aer
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import CircuitQNN

@ddt
class TestQbmNN(QiskitMachineLearningTestCase):
    """QBM QNN Tests."""


@data([73, 9], [72, 8], [0, 0], [1, 1], [24, 0], [56, 0], [2, 2], [64, 16])
    @unpack
    def test_reduce_label(self, label, expected_label):
        """Tests if binary labels are reduced correctly by discarding aux registers."""
        gibbs_state_function = Zero
        hamiltonian = SummedOp(
            [
                0.3 * Z ^ Z ^ I ^ I ^ I ^ I ^ I,
                0.2 * Z ^ I ^ I ^ I ^ I ^ I ^ I,
                0.5 * I ^ Z ^ I ^ I ^ I ^ I ^ I,
            ]
        )
        temperature = 42

        backend = Aer.get_backend("qasm_simulator")

        depth = 1
        num_qubits = 7

        aux_registers = set(range(3, 6))

        ansatz = build_ansatz(num_qubits, depth)
        param_values_init = build_init_ansatz_params_vals(num_qubits, depth)

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))
        gibbs_state = GibbsStateSampler(
            gibbs_state_function,
            hamiltonian,
            temperature,
            backend,
            ansatz,
            params_dict,
            aux_registers=aux_registers,
        )

        label = 73
        reduced_label = gibbs_state._reduce_label(label)
        expected_label = 9
        np.testing.assert_equal(reduced_label, expected_label)