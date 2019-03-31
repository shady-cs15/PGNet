# PGNet
pointer generator reimplementation for CS11747

# Files
**preprocessing**: Scripts used from the author's implementation: preprocessing.py but updated as per need <br />
**batching**: mybatcher.py<br />
**training**: train.py<br />
**decoding**: decode.py<br />
**model**: model.py<br />
**hyperparams**: config.py </br>

# Setting up and running
1. clone <br/>
2. download and preprocess data : follow https://github.com/abisee/cnn-dailymail/issues/9 <br/>
3. training: update config.py and run train.py <br/>
4. evaluation: run decode.py and evaluate with pyrouge </br>
pyrouge setup instructions: http://kavita-ganesan.com/rouge-howto
