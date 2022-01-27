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
"""Quantum Convolutional Neural Network class."""

from typing import List, Optional, Union, Tuple

import numpy as np
from qiskit.circuit import Parameter
from qiskit.opflow import (
    Gradient,
    OperatorBase,
    ExpectationBase,
)
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance

try:
    from sparse import SparseArray
except ImportError:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


from ..neural_network import NeuralNetwork


class QcnnNeuralNetwork(NeuralNetwork):
    """Quantum Convolutional Neural Network class."""

    def __init__(
        self,
        operator: OperatorBase,
        input_params: Optional[List[Parameter]] = None,
        weight_params: Optional[List[Parameter]] = None,
        exp_val: Optional[ExpectationBase] = None,
        gradient: Optional[Gradient] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
        input_gradients: bool = False,
    ):
        """
        Args:
            operator: The parametrized operator that represents the neural network.
            input_params: The operator parameters that correspond to the input of the network.
            weight_params: The operator parameters that correspond to the trainable weights.
            exp_val: The Expected Value converter to be used for the operator.
            gradient: The Gradient converter to be used for the operator's backward pass.
            quantum_instance: The quantum instance to evaluate the network.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using ``TorchConnector``.
        """
        self._input_params = list(input_params) or []
        self._weight_params = list(weight_params) or []
        self._set_quantum_instance(quantum_instance)
        self._operator = operator
        self._forward_operator = exp_val.convert(operator) if exp_val else operator
        self._gradient = gradient
        self._input_gradients = input_gradients
        self._construct_gradient_operator()

        output_shape = self._compute_output_shape(operator)
        super().__init__(
            len(self._input_params),
            len(self._weight_params),
            sparse=False,
            output_shape=output_shape,
            input_gradients=input_gradients,
        )

    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Union[np.ndarray, SparseArray]:
        raise NotImplementedError

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[Union[np.ndarray, SparseArray]], Optional[Union[np.ndarray, SparseArray]],]:
        raise NotImplementedError
