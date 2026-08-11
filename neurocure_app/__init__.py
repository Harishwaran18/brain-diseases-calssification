"""NeuroCure -- interactive web platform for 3D neurodegenerative brain
reconstruction and computational evaluation for neuroregenerative therapy.

Entry point::

    streamlit run neurocure_app/app.py

The app drives a guided multi-page workflow (upload -> 3D brain -> predict ->
therapy -> live cure simulation -> report) on top of the ``brainframe`` engine
:class:`~brainframe.session.Session`.
"""

__version__ = "0.1.0"
