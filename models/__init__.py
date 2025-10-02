# --- START OF models/__init__.py --- (Corrected)

# REMOVE this line: from .ModelBase import ModelBase

def import_model(model_class_name):
    # Keep this function as it was
    module = __import__('Model_'+model_class_name, globals(), locals(), [], 1)
    return getattr(module, 'Model')

# --- END OF models/__init__.py ---