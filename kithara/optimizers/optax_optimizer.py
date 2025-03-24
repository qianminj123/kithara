from typing import Tuple

import jax
from jaxtyping import PyTree
from keras.src import backend
import optax

from kithara.optimizers.protocol import KitharaOptimizer, TrainableVariable, OptimizerVariable


class OptaxOptimizerInterface(KitharaOptimizer):
    def __init__(self, optimizer: optax.GradientTransformation, params: PyTree[backend.Variable]):
        self.optimizer, state = optimizer, optimizer.init(params)
        self.state_in_keras_variable = self._convert_state_to_keras_variables(state, params)

        with backend.name_scope("OptaxOptimizerInterface"):
            iterations = backend.Variable(0, name="iterations")

        self._iterations = iterations

    @property
    def learning_rate(self) -> None:
        return

    @property
    def variables(self) -> PyTree[backend.Variable]:
        return self.state_in_keras_variable

    @property
    def iterations(self) -> backend.Variable:
        return self._iterations

    def stateless_apply(self, optimizer_variable: OptimizerVariable, gradients: PyTree,
                        trainable_variables: TrainableVariable
                        ) -> Tuple[TrainableVariable, OptimizerVariable]:

        updates, opt_state = self.optimizer.update(
            gradients, optimizer_variable, trainable_variables)
        new_trainable_variables = optax.apply_updates(trainable_variables, updates)

        return new_trainable_variables, opt_state

    def get_optimizer_memory_usage(self):
        return sum([leaf.nbytes for leaf in jax.tree_util.tree_leaves(self.variables)])

    def _convert_state_to_keras_variables(
            self, opt_state: PyTree, trainable_vars: PyTree[backend.Variable]) -> PyTree[backend.Variable]:

        _, trainable_tree_def = jax.tree.flatten(trainable_vars)

        def is_same_shape_with_trainable_subtree(opt_state_subtree: PyTree):
            _, opt_state_tree_def = jax.tree.flatten(opt_state_subtree)
            return opt_state_tree_def == trainable_tree_def

        def create_keras_variable_pytree(
                subtree, trainable_vars) -> PyTree[backend.Variable]:

            if not is_same_shape_with_trainable_subtree(subtree):
                return jax.tree.map(
                    lambda leaf: backend.Variable(leaf,), subtree
                )

            return jax.tree.map(
                lambda leaf, keras_var: backend.Variable(
                    leaf, name=f"{keras_var.path.replace('/', '_')}_optstate",),
                subtree,
                trainable_vars)

        return jax.tree.map(
            lambda subtree: create_keras_variable_pytree(subtree, trainable_vars),
            opt_state,
            is_leaf=is_same_shape_with_trainable_subtree
        )
