import numpy as np

# activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

#LSTM
class SimpleLSTMCell():
    def __init__(self, input_size, hidden_size):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # forget Gate weights and bias
        self.Wf = np.random.randn(hidden_size, input_size) * 0.1
        self.Uf = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Bf = np.zeros((hidden_size, 1))
        
        # input Gate weights and bias
        self.Wi = np.random.randn(hidden_size, input_size) * 0.1
        self.Ui = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Bi = np.zeros((hidden_size, 1))
        
        # candidate Cell State weights and bias
        self.Wc = np.random.randn(hidden_size, input_size) * 0.1
        self.Uc = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Bc = np.zeros((hidden_size, 1))
        
        # output Gate weights and bias
        self.Wo = np.random.randn(hidden_size, input_size) * 0.1
        self.Uo = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
              
        f_t = sigmoid(np.dot(self.Wf, x) + np.dot(self.Uf, h_prev) + self.Bf)
        i_t = sigmoid(np.dot(self.Wi, x) + np.dot(self.Ui, h_prev) + self.Bi)
        c_bar = tanh(np.dot(self.Wc, x) + np.dot(self.Uc, h_prev) + self.Bc)
        c_t = (f_t * c_prev) + (i_t * c_bar)
        o_t = sigmoid(np.dot(self.Wo, x) + np.dot(self.Uo, h_prev) + self.Bo)
        h_t = o_t * tanh(c_t)
        
        # save variables in a Dictionary for the backward pass
        cache = {
            'f_t': f_t,
            'i_t': i_t,
            'o_t': o_t,
            'c_t': c_t,
            'c_bar': c_bar,
            'c_prev': c_prev,
            'h_prev': h_prev,
            'x': x         
        }
        
        return h_t, c_t, cache
    
    def backward(self, cache, dh_t, dc_next):
        
        # unpack the dictionary
        f_t, i_t, o_t, c_t, c_bar, c_prev, h_prev, x = cache.values()
        
        # output gate gradients
        do_t = dh_t * tanh(c_t)
        do_raw = do_t * o_t * (1 - o_t)
        
        # cell state gradients
        dc_t = dc_next + (dh_t * o_t * (1 - tanh(c_t)**2))
        
        # forget, input, and candidate gradients
        df_t = dc_t * c_prev
        df_raw = df_t * f_t * (1- f_t)
        
        di_t = dc_t * c_bar
        di_raw = di_t * i_t * (1 - i_t)
        
        dc_bar = dc_t * i_t
        dc_bar_raw = dc_bar * (1 - c_bar**2)
        
        # calculate weight gradients
        dw_f = np.dot(df_raw, x.T)
        du_f = np.dot(df_raw, h_prev.T)
        db_f = df_raw
        
        dw_i = np.dot(di_raw, x.T)
        du_i = np.dot(di_raw, h_prev.T)
        db_i = np.sum(di_raw, axis=1, keepdims=True)
        
        dw_c = np.dot(dc_bar_raw, x.T)
        du_c = np.dot(dc_bar_raw, h_prev.T)
        db_c = np.sum(dc_bar_raw, axis=1, keepdims=True)
        
        dw_o = np.dot(do_raw, x.T)
        du_o = np.dot(do_raw, h_prev.T)
        db_o = np.sum(do_raw, axis=1, keepdims=True)
        
        # gradients with respect to inputs (for previous layers/time steps)
        dx = np.dot(self.Wf.T, df_raw) + np.dot(self.Wi.T, di_raw) + np.dot(self.Wc.T, dc_bar_raw) + np.dot(self.Wo.T, do_raw)
        dh_prev = np.dot(self.Uf.T, df_raw) + np.dot(self.Ui.T, di_raw) + np.dot(self.Uc.T, dc_bar_raw) + np.dot(self.Uo.T, do_raw)
        dc_prev = dc_t * f_t
        
        # group weight gradients in a dictionary
        gradients = {
            'wf':dw_f, 'uf':du_f, 'bf':db_f,
            'wi':dw_i, 'ui':du_i, 'bi':db_i,
            'wc':dw_c, 'uc':du_c, 'bc':db_c,
            'wo':dw_o, 'uo':du_o, 'bo':db_o
        }
        
        return dx, dh_prev, dc_prev, gradients
    
    def update_weights(self, gradients, lr):
        
        # Apply Gradient Descent to update weights
        self.Wf -= lr * gradients['wf']
        self.Uf -= lr * gradients['uf']
        self.Bf -= lr * gradients['bf']
        
        self.Wi -= lr * gradients['wi']
        self.Ui -= lr * gradients['ui']
        self.Bi -= lr * gradients['bi']
        
        self.Wc -= lr * gradients['wc']
        self.Uc -= lr * gradients['uc']
        self.Bc -= lr * gradients['bc']
        
        self.Wo -= lr * gradients['wo']
        self.Uo -= lr * gradients['uo']
        self.Bo -= lr * gradients['bo']

