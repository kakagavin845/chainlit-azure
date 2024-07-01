1. Create a virtualenv and activate it
   ```
   python3 -m venv .venv && source .venv/bin/activate
   ```

   If you have python 3.11, then the above command is fine. But, if you have python version less than 3.11. Using conda is easier. First make sure that you have conda installed. Then run the following command.
   ```
   conda create -n .venv python=3.11 -y && source activate .venv
   ```

2. Run the following command in the terminal to install necessary python packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the following command in your terminal to start the chat UI:
   ```
   chainlit run app.py -w
   ```
