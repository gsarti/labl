from abc import abstractmethod
from collections.abc import Callable
from inspect import isroutine

from wqe.utils.typing import LabelType


class BaseEntry:
    """Base class for all data entries. This class handles the creation of public getters, disallowing setting and
    providing a private constructor key to prevent direct instantiation.
    """

    _label_types: list[type]

    def __init_subclass__(cls, **kwargs):
        """
        Class decorator to automatically create getters for private attributes.
        Setting private attributes is disallowed for children classes.

        Example: An attribute `_name` will get a public property `name`.
        """
        potential_attrs = set(vars(cls).keys())
        if hasattr(cls, "__annotations__"):
            potential_attrs.update(cls.__annotations__.keys())

        private_attr_names = [
            name
            for name in potential_attrs
            if name.startswith("_")
            and not name.startswith("__")
            and not isroutine(getattr(cls, name))  # Exclude helper methods
        ]

        for private_name in private_attr_names:
            public_name = private_name.lstrip("_")

            # Safety check: Don't overwrite something that already exists with the public name
            if hasattr(cls, public_name):
                print(
                    f"Warning: Attribute/method '{public_name}' already exists in class "
                    f"'{cls.__name__}'. Skipping property creation for '{private_name}'."
                )
                continue

            def make_getter(public_name, p_name):
                def getter(self):
                    return getattr(self, p_name)

                getter.__doc__ = f"Getter for the '{public_name}' property, accessing '{p_name}'."
                return getter

            def make_disabled_setter():
                def setter(self, value):
                    raise RuntimeError(
                        f"{cls.__name__} instances cannot be modified after initialization. "
                        f"Create a new instance of '{cls.__name__}' instead."
                    )

                return setter

            prop = property(
                fget=make_getter(public_name, private_name),
                fset=make_disabled_setter(),
                doc=f"Property to access the private attribute '{private_name}'.",
            )
            setattr(cls, public_name, prop)

    ### Helper Functions ###

    @abstractmethod
    def _get_label_types(self) -> list[type]:
        pass

    @abstractmethod
    def _relabel(self, relabel_fn: Callable[[LabelType], LabelType]) -> None:
        pass

    ### Utility Functions ###

    @abstractmethod
    def relabel(
        self,
        relabel_fn: Callable[[LabelType], LabelType] | None = None,
        relabel_map: dict[str | int, LabelType] | None = None,
    ) -> None:
        """Relabels the entry in-place using a custom relabeling function or a mapping.

        Args:
            relabel_fn (Callable[[str | int | float | None], str | int | float | None]):
                A function that will be applied to each label in the entry.
                The function should take a single argument (the label) and return the new label.
                The function should return the label without any processing if the label should be preserved.
            relabel_map (dict[str | int, str | int | float | None]):
                A dictionary that maps old labels to new labels. The keys are the old labels and the values are the
                new labels. This can be used instead of the relabel_fn to relabel the entry if labels are discrete.
        """
        if relabel_fn is None:
            if relabel_map is None:
                raise ValueError("Either relabel_fn or relabel_map must be provided.")
            relabel_fn = lambda x: x if x is None or isinstance(x, float) else relabel_map.get(x, x)
        self._relabel(relabel_fn)
        self._label_types = self._get_label_types()
