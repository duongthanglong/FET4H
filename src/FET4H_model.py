import tensorflow as tf, inspect, io, sys
#@title ⭐⭐⭐Create new MODEL from designing ARCHITECTUTRES on the DATASET
#==============================================================================#
class MyModel():
    expand_dims = tf.expand_dims
#______________________________________________________________________________#
    def basic_block(x, filters, kernel, stride, padding, order, bias): #block: default order = conv+batch_norm+activation, if cb is conv+batch_norm
        opt = {
            'c': tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding = padding, use_bias=bias),
            'b': tf.keras.layers.BatchNormalization(),
            'a': tf.keras.layers.Activation('relu')
        }
        for o in order: x = opt[o](x)
        return x
#______________________________________________________________________________#
    def residual_block(x, filters, kernel, stride, bias, downsamples): #residual block
        shortcut = x
        x = MyModel.basic_block(x, filters, kernel, stride, 'same','cba',bias)
        x = MyModel.basic_block(x, filters, kernel, stride, 'same','cb',bias)
        if downsamples or stride!=1 or tf.shape(shortcut)[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=bias)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x
#______________________________________________________________________________#
    def dense_block(x, filters, repetition): #repetition = number of component [2xBAC layers] in dense block
        for _ in range(repetition):
            y = MyModel.basic_block(x, filters, 1, 1, 'same', 'bac', True)
            y = MyModel.basic_block(y, filters//4, 3, 1, 'same', 'bac', True)
            x = tf.keras.layers.concatenate([y,x])
        return x
#______________________________________________________________________________#
    def transition_block(x, filters): #transition block
        x = tf.keras.layers.Conv2D(filters, 1, 1, padding = 'same', activation='relu')(x)
        x = tf.keras.layers.AvgPool2D(2, 2, padding = 'same')(x)
        return x
#______________________________________________________________________________#
    def squeeze_excitation(input_feature, ratio): #ratio=16
        filters = input_feature.shape[-1] #tf.shape(input_feature)[-1]
        se = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
        se = tf.keras.layers.Reshape((1, 1, filters))(se)
        se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
        se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
        return tf.keras.layers.multiply([input_feature, se])
#______________________________________________________________________________#
    def channel_attention(input_feature, ratio): #channels=64, ratio=8

        filters = input_feature.shape[-1]
        fc1 = tf.keras.layers.Dense(filters//ratio, activation='relu')
        fc2 = tf.keras.layers.Dense(filters)

        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
        avg_pool = MyModel.expand_dims(MyModel.expand_dims(avg_pool, 1), 1)
        avg_pool = fc2(fc1(avg_pool))

        max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
        max_pool = MyModel.expand_dims(MyModel.expand_dims(max_pool, 1), 1)
        max_pool = fc2(fc1(max_pool))

        attention_feature = tf.keras.layers.Add()([avg_pool,max_pool])
        attention_feature = tf.keras.layers.Activation('sigmoid')(attention_feature)
        return tf.keras.layers.multiply([input_feature, attention_feature])
#______________________________________________________________________________#
    def spatial_attention(input_feature):
        conv1 = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        reduce_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))
        reduce_max = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))
        avg_pool = reduce_mean(input_feature)
        max_pool = reduce_max(input_feature)
        concat = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))
        attention_feature = concat([avg_pool, max_pool])
        attention_feature = conv1(attention_feature)
        return tf.keras.layers.multiply([input_feature, attention_feature])
#______________________________________________________________________________#
    def fusion_spatialchannel_attention(input_feature, ratio, weight_channel): # weight_channel = None is sequential fusion
        cam_feature = MyModel.channel_attention(input_feature, ratio)
        if weight_channel is None:
            sam_feature = MyModel.spatial_attention(cam_feature)
            return sam_feature #sequential of channel, spatial attention (CSAM)
        elif weight_channel==1:
            return cam_feature
        else:
            sam_feature = MyModel.spatial_attention(input_feature)
            att_feature = tf.keras.layers.Add()([cam_feature*weight_channel, sam_feature*(1-weight_channel)])
            return att_feature #weighted fusion of channel & spatial attention (wCSAM)
#______________________________________________________________________________#
#______________________________________________________________________________#
    #Define a sub_class for GRN
    @tf.keras.utils.register_keras_serializable(package="MyLayers")#tf.keras.saving.register_keras_serializable(package="MyLayers")
    class GRN(tf.keras.layers.Layer):
        def __init__(self, dim, name='grn', **kwargs):
            super().__init__(**kwargs)
            self.dim = dim
            self.var_name = name
        def build(self, input_shape):
            self.gamma = tf.Variable(
                initial_value=tf.zeros([1, 1, 1, self.dim], dtype=self.compute_dtype),
                trainable=True, dtype=self.compute_dtype, name=f'{self.var_name}/gamma'
            )
            self.beta = tf.Variable(
                initial_value=tf.zeros([1, 1, 1, self.dim], dtype=self.compute_dtype),
                trainable=True, dtype=self.compute_dtype, name=f'{self.var_name}/beta'
            )
        def call(self, x):
            Gx = tf.norm(x, ord='euclidean', axis=(1,2), keepdims=True)
            Nx = Gx / (tf.math.reduce_mean(Gx, axis=-1, keepdims=True) + 1e-6)
            return self.gamma * (x * Nx) + self.beta + x
        def get_config(self):
            config = super(MyModel.GRN, self).get_config()
            config.update({ 'dim': self.dim, 'name': self.var_name })
            return config
        @classmethod
        def from_config(cls, config):
            return cls(**config)
#______________________________________________________________________________#
    @tf.keras.utils.register_keras_serializable(package="MyLayers")#tf.keras.saving.register_keras_serializable(package="MyLayers")
    class FusedAttention(tf.keras.layers.Layer):
        def __init__(self, ratio, name='attention', **kwargs):
            super().__init__(**kwargs)
            self.ratio = ratio
            self.var_name = name
        def build(self, input_shape):
            self.weight_cs = tf.Variable(
                shape=[1], initial_value=tf.constant([0.5], dtype=tf.float32),
                trainable=True, dtype=tf.float32, name=f'{self.var_name}/weight_cs',
                constraint=tf.keras.constraints.MinMaxNorm(min_value=0., max_value=1.)
            )
            filters = input_shape[-1]
            self.fc1 = tf.keras.layers.Dense(filters//self.ratio, activation='relu')
            self.fc2 = tf.keras.layers.Dense(filters)
            self.conv1 = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        def call(self, x):
            #channel attention
            avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
            avg_pool = MyModel.expand_dims(MyModel.expand_dims(avg_pool, 1), 1)
            avg_pool = self.fc2(self.fc1(avg_pool))
            max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
            max_pool = MyModel.expand_dims(MyModel.expand_dims(max_pool, 1), 1)
            max_pool = self.fc2(self.fc1(max_pool))
            ca = tf.keras.layers.Add()([avg_pool,max_pool])
            ca = tf.keras.layers.Activation('sigmoid')(ca)
            #spatial attention
            reduce_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))
            reduce_max = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))
            avg_pool = reduce_mean(x)
            max_pool = reduce_max(x)
            sa = tf.keras.layers.concatenate([avg_pool, max_pool])
            sa = self.conv1(sa)
            #weighted attention
            w = self.weight_cs[0]*ca + (1-self.weight_cs[0])*sa
            x = tf.keras.layers.multiply([x, w])
            return x
        def get_config(self):
            config = super(MyModel.FusedAttention, self).get_config()
            config.update({ 'ratio': self.ratio, 'name': self.var_name })
            return config
        @classmethod
        def from_config(cls, config):
            return cls(**config)
#______________________________________________________________________________#
    # Define a downsample
    def downsample_block(dim, kernel, stride, order='cn'):
        '''1: conv2d->norm / 2: norm->conv2d'''
        conv2d = tf.keras.layers.Conv2D(dim, kernel_size=kernel, strides=stride, padding='valid')
        norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        opts = [conv2d,norm] if order=='cn' else [norm,conv2d]
        def downsample_fn(x):
            #x = MyModel.multihead_attention_block(x,num_heads=8)
            x = opts[0](x)
            x = opts[1](x)
            return x
        return downsample_fn
#______________________________________________________________________________#
    def multihead_attention_block(x, dim, num_heads, key_dim, dropout):
        ''' dim = last dimension of X (channels) '''
        b,h,w,c = tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3]
        x = tf.reshape(x,(b,h*w,c))
        # Input projections
        q = tf.keras.layers.Dense(num_heads*key_dim, use_bias=False)(x)
        k = tf.keras.layers.Dense(num_heads*key_dim, use_bias=False)(x)
        v = tf.keras.layers.Dense(num_heads*key_dim, use_bias=False)(x)
        # Multi-Head Attention
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        ao = mha(q, k, v)
        ao = tf.keras.layers.Dropout(dropout)(ao)
        ao = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ao)
        # Feed-Forward Network
        fo = tf.keras.layers.Dense(dim*4, activation="relu")(ao)
        fo = tf.keras.layers.Dense(dim)(fo)
        fo = tf.keras.layers.Dropout(dropout)(fo)
        op = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ao + fo)

        output = tf.reshape(op, (b, h, w, c))
        return  output
#______________________________________________________________________________#
    # Define a block
    def FAR_block(dim, drop_path):
        def apply(x):
            input_x = x
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
            x = MyModel.FusedAttention(ratio=8)(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            x = tf.keras.layers.Dense(4 * dim)(x)
            x = tf.keras.layers.Activation("gelu")(x)
            x = MyModel.GRN(4 * dim)(x)
            x = tf.keras.layers.Dense(dim)(x)
            if drop_path > 0.:
                x = tf.keras.layers.Dropout(drop_path)(x) #noise_shape=(None, 1, 1, 1)
            x = tf.keras.layers.Add()([input_x, x]) #residual/shortcut connection
            return x
        return apply
#==============================================================================#
    # Define the Modified ConvNeXtV2 model: depths=[number of blocks in stages], dims=[number of output-channels in stages]
    def create_FET4H(given_inputs=None, **input_shape_num_classes_downsamples_depths_dims_drop_path_rate):
        '''downsamples=[stem:(4-kernel, 4-stride), others:(-,-),(-,-),(-,-)], depths=[2,2,6,2], dims=[40,80,160,320]'''        
        ic4d = input_shape_num_classes_downsamples_depths_dims_drop_path_rate
        if ic4d is None:
            ic4d = {  'input_shape': [70,70,3],
                      'num_classes': 3,
                      'downsamples': [(7,2),(2,2),(2,2),(2,2)],
                      'depths': [2,2,2,2],
                      'dims': [40,80,160,320],
                      'drop_path_rate': 0.25  }
        input_shape,num_classes = ic4d['input_shape'],ic4d['num_classes']
        downsamples,depths,dims,drop_path_rate = ic4d['downsamples'],ic4d['depths'],ic4d['dims'],ic4d['drop_path_rate']

        inputs = tf.keras.layers.Input(shape=input_shape) if given_inputs is None else given_inputs
        x = inputs

        dp_rates = [d.numpy() for d in tf.linspace(0., drop_path_rate, sum(depths))]
        cur = 0

        # Stages (0=Stem,1,2,...): downsample + blocks
        for i in range(len(depths)):
            dim = dims[i]
            kernel,stride = downsamples[i]
            x = MyModel.downsample_block(dim, kernel, stride, order='cn' if i==0 else 'nc')(x)
            for j in range(depths[i]): #blocks
                x = MyModel.FAR_block(dim, dp_rates[cur])(x)
                cur += 1

        # Final norm layer
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        if given_inputs is not None: return x
        # Classification head
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        fname = inspect.getframeinfo(inspect.currentframe()).function
        return tf.keras.models.Model(inputs, outputs, name=fname.split('_')[-1])
#==============================================================================#
    def print_model(model, firstlines=5, lastlines=None):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        if lastlines is not None:
            summary_lines = stream.getvalue().splitlines()
            summary_lines = '\n'.join(summary_lines[:firstlines]+['.'*len(summary_lines[1])]+summary_lines[-lastlines:])
        else: summary_lines = stream.getvalue()
        print(summary_lines)
#==============================================================================#
MyModel.ListModel = { 0: MyModel.create_FET4H }
#______________________________________________________________________________#
if __name__ == "__main__":
    kModel = 0
    if kModel==0:
        pModel = {'input_shape': [70,70,3],
                  'num_classes': 3,
                  'downsamples': [(7,2),(2,2),(2,2),(2,2)],
                  'depths': [2,2,2,2],
                  'dims': [40,80,160,320],
                  'drop_path_rate': 0.25 }
    model = MyModel.ListModel[kModel](**pModel)
    MyModel.print_model(model,5,15)
