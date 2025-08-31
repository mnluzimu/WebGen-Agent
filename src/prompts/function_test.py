function_test_prompt = """Based on the original website development instruction, you should identify the requirements and create an instruction for a web-navigation GUI agent to test the generated website. The following is an example of triggering the GUI agent testing based on the original instruction:

<example>
Original instruction: Please implement a self-driving tour website that provides self-driving tour products and services. The website should have functionalities for browsing self-driving tour routes, booking self-driving tour hotels, and self-help self-driving tour packages. Users should be able to browse different types of self-driving tour routes, book hotels and packages, and query self-driving club information. The website should also provide search and filtering functions to help users quickly find the self-driving tour products they need. Define background as cream; define components with dark teal.

<boltAction type="gui_agent_test">Verify cream background and dark‑teal buttons. Browse different types of self-driving tour routes, book hotels and packages, and query self-driving club information. Search and filter for self-driving tour products.</boltAction>
</example>

The following is the original website developemnt instruction:

<instruction>{instruction}</instruction>

Trigger the GUI agent testing based on the original instruction in a way similar to the example. Do not generate additional comments.
"""