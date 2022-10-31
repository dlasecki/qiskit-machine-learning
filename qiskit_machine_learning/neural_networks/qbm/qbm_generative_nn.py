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
from qiskit_machine_learning.datasets.dataset_helper import discretize_and_truncate

from qiskit_machine_learning.algorithms import TrainableModel

from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss, Loss

from qiskit_machine_learning.neural_networks.qbm.qbm_nn import QbmNN


class QbmGenerativeNN(ABC, QbmNN, TrainableModel):
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
            loss: Loss = CrossEntropyLoss,
            input_gradients: bool = False,
            init_params: Optional[Dict[Parameter, complex]] = None,
            backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        super(QbmNN).__init__(gibbs_state_builder, temperature, visible_units, param_hamiltonian,
                              num_inputs, num_weights, sparse, output_shape, input_gradients,
                              init_params, backend)
        super(TrainableModel).__init__()

        self._loss = loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrainableModel":
        pass

    def _calc_p_v_from_data(self, train_data) -> np.ndarray:
        probs = discretize_and_truncate(train_data, return_prob=True)

    # # TODO move to child classes, will use existing LossFunction impl
    # def calc_obj_fun_grad(
    #         self,
    #         p_v_data: np.ndarray,
    #         measurement_op: OperatorBase,
    #         temperature: float,
    #         hamiltonian_param_dict: Dict[Parameter, float],
    #         gradient_method: str = "param_shift",
    # ) -> Dict[Parameter, float]:
    #     gradient_params = self._param_hamiltonian.ordered_parameters  # validate with Gibbs state
    #
    #     gibbs_state = self._gibbs_state_builder.build(self._param_hamiltonian, temperature)
    #     p_v_qbm = self._calc_p_v_qbm(
    #         gibbs_state, hamiltonian_param_dict
    #     )  # H params should be bound already
    #     gibbs_ham_gradients = gibbs_state.calc_hamiltonian_gradients(
    #         gradient_params, measurement_op, gradient_method
    #     )  # H params should be bound already
    #
    #     obj_fun_grads = {}
    #     # TODO having them as an ordered np.ndarray would allow using sum which could be faster
    #     for hamiltonian_param in gibbs_ham_gradients.keys():
    #         gradient = -p_v_data / p_v_qbm * gibbs_ham_gradients[hamiltonian_param]
    #         obj_fun_grads[hamiltonian_param] = gradient
    #
    #     return obj_fun_grads
