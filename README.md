## How to run with jupyter notebook

For newer version of jupyter, you might get an error 

``IOPub data rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_data_rate_limit`.``

To fix this, run with 

``jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000.``