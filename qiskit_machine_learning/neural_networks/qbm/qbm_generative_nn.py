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

from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.neural_networks.qbm.qbm_nn import QbmNN


class QbmGenerativeNN(ABC, QbmNN):
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
        super().__init__(gibbs_state_builder, temperature, visible_units, param_hamiltonian,
                         num_inputs, num_weights, sparse, output_shape, input_gradients,
                         init_params, backend)
