class InvarianceSpecification:
    _no_invariance_keys = {
        None,
        "none",
        "no_invariance"
    }

    _shift_invariance_keys = {
        "shift",
        "1d_shift",
        "shift_invariance",
        "1d_shift_invariance"
    }

    _translation_invariance_keys = {
        "translation",
        "2d_translation",
        "translation_invariance",
        "2d_translation_invariance"
    }

    @staticmethod
    def no_invariance(invariance_type: str) -> bool:
        return invariance_type in InvarianceSpecification._no_invariance_keys

    @staticmethod
    def shift_invariance(invariance_type: str) -> bool:
        return invariance_type in InvarianceSpecification._shift_invariance_keys

    @staticmethod
    def translation_invariance(invariance_type: str) -> bool:
        return invariance_type in InvarianceSpecification._translation_invariance_keys


__all__ = ["InvarianceSpecification"]
