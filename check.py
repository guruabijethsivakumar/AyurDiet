from owlready2 import get_ontology, default_world
import os

# Path to your OWL file
owl_file_path = "dosha_ontology.owl"  # Replace with your file path
pellet_path = "C:\\sih\\pellet\\pellet-owlapi-ignazio1977-2.4.0-ignazio1977.jar"  # Update with the downloaded file path

try:
    # Load the ontology
    onto = get_ontology(owl_file_path).load()

    print("OWL file loaded successfully. Performing validation...")

    # Configure and sync reasoner
    with onto:
        reasoning_successful = False
        if os.path.exists(pellet_path):
            print("Using Pellet reasoner with custom path.")
            onto.get_backend().reasoner = "pellet"
            onto.get_backend().reasoner_path = pellet_path
            try:
                onto.sync_reasoner(infer_property_values=True)
                reasoning_successful = True
                print("Reasoner sync completed.")
            except Exception as reasoner_error:
                print(f"Pellet error: {str(reasoner_error)}")
        else:
            print("Pellet jar not found. Proceeding without full reasoning.")

        # Check for unsatisfiable classes only if reasoning was successful
        if reasoning_successful:
            unsatisfiable_classes = [cls for cls in onto.classes() if cls.is_unsatisfiable()]
            if unsatisfiable_classes:
                print("Warning: Unsatisfiable classes found:")
                for cls in unsatisfiable_classes:
                    print(f"- {cls.name}")
            else:
                print("No unsatisfiable classes detected. Ontology appears logically consistent.")
        else:
            print("Reasoning not performed due to missing reasoner. Skipping satisfiability check.")

        # Check for undefined entities (does not require reasoning)
        undefined_entities = []
        for entity in onto.entities():
            for prop in entity.get_properties():
                for value in prop[entity]:
                    if value not in onto.entities():
                        undefined_entities.append((entity, prop, value))
        if undefined_entities:
            print("Warning: Undefined entities referenced:")
            for entity, prop, value in undefined_entities:
                print(f"- Entity: {entity.name}, Property: {prop.name}, Value: {value}")
        else:
            print("No undefined entities detected.")

        print("Validation completed.")

except Exception as e:
    print(f"Error during validation: {str(e)}")
    print("Please check the OWL file for syntax errors, missing declarations, or reasoner configuration.")