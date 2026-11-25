"""Controllers: the only layer the frontend talks to.

They validate user input, call the backend, and return plain data (dicts / DataFrames)
ready for display. Model loading is cached here with st.cache_resource so each model is
loaded once per server process, not on every click.
"""
