import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
import os
from tqdm import tqdm
from tiny_imagenet import TinyImagenetDataset # https://github.com/ksachdeva/tiny-imagenet-tfds


def train_agilenn_cifar10(
    model,
    evaluator,
    run_name,
    logdir,
    split_ratio=0.2,
    rho=0.8,
    klambda=0.8,):
    ds = tfds.load('cifar10', as_supervised=True)
    
    std = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
    mean= tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))
    
    def train_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - mean) / std
        x = tf.image.resize(x, [96, 96])
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = (x - mean) / std
        x = tf.image.resize(x, [96, 96])
        return x, y
    
    ds_train = ds['train'].map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(128)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                 
    ds_test = ds['test'].map(valid_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(128*4)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
                            
    lr = 1e-1
    weight_decay = 5e-4
    epochs = 200
    decay_steps = len(tfds.as_numpy(ds_train)) * epochs
    
    lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
    wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
    optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    runid = run_name + '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    skewness = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    
    evaluator.trainable = False
    K_top = int(np.round(split_ratio * 24)) # 本地特征通道数
    
    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False) # 这是预训练的特征提取器？
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2: # baseline
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0)) # 求取 梯度和
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2])
            top_importance_scores = per_channel_importance[:, :K_top]
            bottom_importance_scores = per_channel_importance[:, K_top:]
            
            # disorder loss
            total_importance = tf.reduce_sum(per_channel_importance, axis=-1)
            mins = tf.reduce_min(I[:, :, :, :K_top], axis=[1,2,3]) # (None,)
            maxs = tf.reduce_max(I[:, :, :, K_top:], axis=[1,2,3]) # (None,)
            disorder_loss = tf.math.reduce_mean(tf.math.maximum((maxs - mins) / total_importance, 0.0))


            # skewness loss
            local_importance = tf.reduce_sum(top_importance_scores, axis=-1)
            remote_importance = tf.reduce_sum(bottom_importance_scores, axis=-1)
            skewness_list = local_importance / (remote_importance + local_importance + 1e-9)
            achieved_skewness = tf.reduce_mean(skewness_list)
            skewness_loss = tf.reduce_mean(tf.maximum(rho - skewness_list, 0.0), axis=0)
            
            top_f, bottom_f = model.feature_splitter(f)
            bottom_f_q, _ = model.q_layer(bottom_f)
            local_outs = model.local_predictor_1(top_f, training=training)
            remote_outs = model.remote_predictor_1(bottom_f_q, training=training)
            y_pred = model.reweighting_1(local_outs, remote_outs)
            ce_loss = loss_fn_cls(y, y_pred)
            
            loss = klambda * ce_loss + (1 - klambda) * (skewness_loss + disorder_loss)
            
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))     

        accuracy(y, y_pred)
        cls_loss(ce_loss)
        skewness(achieved_skewness)

    training_step = 0
    best_validation_acc = 0
    
    for epoch in range(epochs):
        
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1
            
            step(x, y, training=True)
            
            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/skewness', skewness.result(), training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                    skewness.reset_states()
        
        
        cls_loss.reset_states()
        accuracy.reset_states()
        skewness.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)
            tf.summary.scalar('test/skewness', skewness.result(), training_step)
            print("=================================")
            print("accuracy: ", accuracy.result())
            print("skewness: ", skewness.result())
            print("=================================")
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                
            cls_loss.reset_states()
            accuracy.reset_states()
            skewness.reset_states()


def train_agilenn_cifar100(
    model,
    evaluator,
    run_name,
    logdir,
    split_ratio=0.2,
    rho=0.8,
    klambda=0.8,):
    ds = tfds.load('cifar100', as_supervised=True)
    
    std = tf.reshape((0.267, 0.256, 0.276), shape=(1, 1, 3))
    mean= tf.reshape((0.507, 0.487, 0.441), shape=(1, 1, 3))
    
    def train_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - mean) / std
        x = tf.image.resize(x, [96, 96])
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = (x - mean) / std
        x = tf.image.resize(x, [96, 96])
        return x, y
    
    ds_train = ds['train'].map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(64)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                 
    ds_test = ds['test'].map(valid_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(64*4)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
                            
    lr = 1e-1
    weight_decay = 5e-4
    epochs = 200
    decay_steps = len(tfds.as_numpy(ds_train)) * epochs
    
    lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
    wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
    optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    runid = run_name + '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    skewness = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    
    evaluator.trainable = False
    K_top = int(np.round(split_ratio * 24))
    
    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2])
            top_importance_scores = per_channel_importance[:, :K_top]
            bottom_importance_scores = per_channel_importance[:, K_top:]
            
            # disorder loss
            total_importance = tf.reduce_sum(per_channel_importance, axis=-1)
            mins = tf.reduce_min(I[:, :, :, :K_top], axis=[1,2,3]) # (None,)
            maxs = tf.reduce_max(I[:, :, :, K_top:], axis=[1,2,3]) # (None,)
            disorder_loss = tf.math.reduce_mean(tf.math.maximum((maxs - mins) / total_importance, 0.0))


            # skewness loss
            local_importance = tf.reduce_sum(top_importance_scores, axis=-1)
            remote_importance = tf.reduce_sum(bottom_importance_scores, axis=-1)
            skewness_list = local_importance / (remote_importance + local_importance + 1e-9)
            achieved_skewness = tf.reduce_mean(skewness_list)
            skewness_loss = tf.reduce_mean(tf.maximum(rho - skewness_list, 0.0), axis=0)
            
            top_f, bottom_f = model.feature_splitter(f)
            bottom_f_q, _ = model.q_layer(bottom_f)
            local_outs = model.local_predictor_1(top_f, training=training)
            remote_outs = model.remote_predictor_1(bottom_f_q, training=training)
            y_pred = model.reweighting_1(local_outs, remote_outs)
            ce_loss = loss_fn_cls(y, y_pred)
            
            loss = klambda * ce_loss + (1 - klambda) * (skewness_loss + disorder_loss)
            
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))     

        accuracy(y, y_pred)
        cls_loss(ce_loss)
        skewness(achieved_skewness)

    training_step = 0
    best_validation_acc = 0
    
    for epoch in range(epochs):
        
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1
            
            step(x, y, training=True)
            
            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/skewness', skewness.result(), training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                    skewness.reset_states()
        
        
        cls_loss.reset_states()
        accuracy.reset_states()
        skewness.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)
            tf.summary.scalar('test/skewness', skewness.result(), training_step)
            print("=================================")
            print("accuracy: ", accuracy.result())
            print("skewness: ", skewness.result())
            print("=================================")
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                
            cls_loss.reset_states()
            accuracy.reset_states()
            skewness.reset_states()


def train_agilenn_svhn(
    model,
    evaluator,
    run_name,
    logdir,
    split_ratio=0.2,
    rho=0.8,
    klambda=0.8,):
    ds = tfds.load('svhn_cropped', as_supervised=True)
    
    def train_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = 2 * x - 1
        x = tf.image.resize(x, [96, 96])
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = 2 * x - 1
        x = tf.image.resize(x, [96, 96])
        return x, y
    
    ds_train = ds['train'].map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(64)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                 
    ds_test = ds['test'].map(valid_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(64*4)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
                            
    lr = 1e-1
    weight_decay = 5e-4
    epochs = 200
    decay_steps = len(tfds.as_numpy(ds_train)) * epochs
    
    lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
    wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
    optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    runid = run_name + '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    skewness = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    
    evaluator.trainable = False
    K_top = int(np.round(split_ratio * 24))
    
    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            
            f = model.feature_extractor_1(x, training=training)
            
            with tf.GradientTape() as tape_inner1:
                tape_inner1.watch(f)
                y_pro = evaluator(f, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads = tape_inner1.gradient(c_loss_pro, f)
            
            f0 = tf.zeros_like(f)
            
            with tf.GradientTape() as tape_inner2:
                tape_inner2.watch(f0)
                y_pro = evaluator(f0, training=False)
                c_loss_pro = loss_fn_cls(y, y_pro)
            grads0 = tape_inner2.gradient(c_loss_pro, f0)
            
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2])
            top_importance_scores = per_channel_importance[:, :K_top]
            bottom_importance_scores = per_channel_importance[:, K_top:]
            
            # disorder loss
            total_importance = tf.reduce_sum(per_channel_importance, axis=-1)
            mins = tf.reduce_min(I[:, :, :, :K_top], axis=[1,2,3]) # (None,)
            maxs = tf.reduce_max(I[:, :, :, K_top:], axis=[1,2,3]) # (None,)
            disorder_loss = tf.math.reduce_mean(tf.math.maximum((maxs - mins) / total_importance, 0.0))


            # skewness loss
            local_importance = tf.reduce_sum(top_importance_scores, axis=-1)
            remote_importance = tf.reduce_sum(bottom_importance_scores, axis=-1)
            skewness_list = local_importance / (remote_importance + local_importance + 1e-9)
            achieved_skewness = tf.reduce_mean(skewness_list)
            skewness_loss = tf.reduce_mean(tf.maximum(rho - skewness_list, 0.0), axis=0)
            
            top_f, bottom_f = model.feature_splitter(f)
            bottom_f_q, _ = model.q_layer(bottom_f)
            local_outs = model.local_predictor_1(top_f, training=training)
            remote_outs = model.remote_predictor_1(bottom_f_q, training=training)
            y_pred = model.reweighting_1(local_outs, remote_outs)
            ce_loss = loss_fn_cls(y, y_pred)
            
            loss = klambda * ce_loss + (1 - klambda) * (skewness_loss + disorder_loss)
            
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))     

        accuracy(y, y_pred)
        cls_loss(ce_loss)
        skewness(achieved_skewness)

    training_step = 0
    best_validation_acc = 0
    
    for epoch in range(epochs):
        
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1
            
            step(x, y, training=True)
            
            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()
                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/skewness', skewness.result(), training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                    skewness.reset_states()
        
        
        cls_loss.reset_states()
        accuracy.reset_states()
        skewness.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)
            tf.summary.scalar('test/skewness', skewness.result(), training_step)
            print("=================================")
            print("accuracy: ", accuracy.result())
            print("skewness: ", skewness.result())
            print("=================================")
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                
            cls_loss.reset_states()
            accuracy.reset_states()
            skewness.reset_states()


def train_agilenn_imagenet200(
    model,
    evaluator,
    run_name,
    logdir,
    split_ratio=0.2,
    rho=0.8,
    klambda=0.8,):
    tiny_imagenet_builder = TinyImagenetDataset()
    tiny_imagenet_builder.download_and_prepare(download_dir='../dataset1')
    train_dataset = tiny_imagenet_builder.as_dataset(split="train")
    test_dataset = tiny_imagenet_builder.as_dataset(split="validation")
    
    def train_prep(sample):
        x, y = sample["image"], sample["label"]
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = 2 * x - 1
        x = tf.image.resize(x, [96, 96]) # 128 128
        return x, y

    def valid_prep(sample):
        x, y = sample["image"], sample["label"]
        x = tf.cast(x, tf.float32)/255.
        x = 2 * x - 1
        x = tf.image.resize(x, [96, 96]) # 128 128
        return x, y


    ds_train = train_dataset.map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                                 .shuffle(1024)\
                                 .batch(64)\
                                 .prefetch(buffer_size=tf.data.AUTOTUNE)
                                 
    ds_test = test_dataset.map(valid_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(64*4)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
                            
    lr = 1e-1
    weight_decay = 5e-4
    epochs = 200
    decay_steps = len(tfds.as_numpy(ds_train)) * epochs
    
    lr_schedule = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
    wd_schedule = tf.keras.experimental.CosineDecay(lr * weight_decay, decay_steps=decay_steps)
    optimizer = tfa.optimizers.SGDW(learning_rate=lr_schedule, weight_decay=wd_schedule, momentum=0.9, nesterov=False)
    
    loss_fn_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    runid = run_name + '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    local_accuracy = tf.metrics.SparseCategoricalAccuracy()
    remote_accuracy = tf.metrics.SparseCategoricalAccuracy()

    cls_loss = tf.metrics.Mean()
    skewness = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    
    evaluator.trainable = False
    K_top = int(np.round(split_ratio * 24))

    # 定义一个带有tf.function装饰器的函数，用于实现模型的单步训练或测试过程。
    @tf.function
    def step(x, y, training):
        # 创建一个梯度记录器，用于记录模型的可训练权重的梯度。
        with tf.GradientTape() as tape:

            # 调用模型的feature_extractor_1属性，提取输入图像的特征。
            f = model.feature_extractor_1(x, training=training)
            # 创建一个内部梯度记录器，用于记录特征图的梯度。
            with tf.GradientTape() as tape_inner1:
                # 将特征图作为观察对象。
                tape_inner1.watch(f)
                # 调用评估器，对特征图进行评估，并返回评估结果。
                y_pro = evaluator(f, training=False)
                # 计算评估结果和真实标签之间的交叉熵损失。
                c_loss_pro = loss_fn_cls(y, y_pro)
            # 根据损失和特征图，计算特征图的梯度。
            grads = tape_inner1.gradient(c_loss_pro, f)
            # 创建一个全零的特征图，形状和原始特征图相同。
            f0 = tf.zeros_like(f)
            # 创建另一个内部梯度记录器，用于记录全零特征图的梯度。
            with tf.GradientTape() as tape_inner2:
                # 将全零特征图作为观察对象
                tape_inner2.watch(f0)
                # 调用评估器，对全零特征图进行评估，并返回评估结果。
                y_pro = evaluator(f0, training=False)
                # 计算评估结果和真实标签之间的交叉熵损失。
                c_loss_pro = loss_fn_cls(y, y_pro)
            # 根据损失和全零特征图，计算全零特征图的梯度。
            grads0 = tape_inner2.gradient(c_loss_pro, f0)

            # 计算原始特征图和全零特征图之间的重要性分数，使用公式I = |0.5 * (grads0 + grads) * (f - f0)|。
            I = tf.abs(0.5 * (grads0 + grads) * (f - f0))
            # 对重要性分数在通道维度上求和，得到每个通道的重要性分数，然后将其分割为上半部分和下半部分。
            per_channel_importance = tf.reduce_sum(I, axis=[1, 2])
            top_importance_scores = per_channel_importance[:, :K_top]
            bottom_importance_scores = per_channel_importance[:, K_top:]
            
            # disorder loss
            # 计算无序损失，使用公式disorder_loss = mean(max((maxs - mins) / total_importance, 0.0))，
            # 其中maxs是下半部分重要性分数的最大值，mins是上半部分重要性分数的最小值，total_importance是每个样本的总重要性分数。
            total_importance = tf.reduce_sum(per_channel_importance, axis=-1)
            mins = tf.reduce_min(I[:, :, :, :K_top], axis=[1,2,3]) # (None,)
            maxs = tf.reduce_max(I[:, :, :, K_top:], axis=[1,2,3]) # (None,)
            disorder_loss = tf.math.reduce_mean(tf.math.maximum((maxs - mins) / total_importance, 0.0))


            # skewness loss

            # 计算偏斜度损失，使用公式skewness_loss = mean(max(rho - skewness_list, 0.0))，
            # 其中skewness_list是每个样本的本地重要性与总重要性之比，rho是一个超参数，表示期望的偏斜度阈值。
            local_importance = tf.reduce_sum(top_importance_scores, axis=-1)
            remote_importance = tf.reduce_sum(bottom_importance_scores, axis=-1)
            skewness_list = local_importance / (remote_importance + local_importance + 1e-9)
            achieved_skewness = tf.reduce_mean(skewness_list)
            skewness_loss = tf.reduce_mean(tf.maximum(rho - skewness_list, 0.0), axis=0)

            # 调用模型的feature_splitter方法，将特征图分割为上半部分和下半部分。
            top_f, bottom_f = model.feature_splitter(f)
            bottom_f_q, _ = model.q_layer(bottom_f)

            local_outs = model.local_predictor_1(top_f, training=training)
            remote_outs = model.remote_predictor_1(bottom_f_q, training=training)
            # 调用模型的reweighting_1属性，对本地和远程预测结果进行加权融合，并返回最终预测结果。

            y_pred = model.reweighting_1(local_outs, remote_outs)

            ce_loss = loss_fn_cls(y, y_pred)
            
            loss = klambda * ce_loss + (1 - klambda) * (skewness_loss + disorder_loss)
        # 如果是训练状态，根据总损失和模型的可训练权重，计算梯度，并使用优化器更新权重。
        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        # 计算分类准确率、分类损失和聚类偏斜度等指标，并更新对应的tf.metrics对象。
        accuracy(y, y_pred)
        local_accuracy(y,local_outs)
        remote_accuracy(y, remote_outs)
        cls_loss(ce_loss)
        skewness(achieved_skewness)

    training_step = 0
    best_validation_acc = 0
    
    for epoch in range(epochs):
        
        for x, y in tqdm(ds_train, desc=f'epoch {epoch+1}/{epochs}', ascii=True):

            training_step += 1
            

            if (epoch == 0):
                for x, y in ds_test:
                    step(x, y, training=False)
                print("accuracy: ", accuracy.result())
                print("local_accuracy: ", local_accuracy.result())
                print("remote_accuracy: ", remote_accuracy.result())
                accuracy.reset_states()
                local_accuracy.reset_states()
                remote_accuracy.reset_states()
                model.load_weights('saved_models/agilenn_mobilenetv2_imagenet200_x7424.tf')
                print("Successfully load")
                for x, y in ds_test:
                    step(x, y, training=False)
                print("accuracy: ", accuracy.result())
                print("local_accuracy: ", local_accuracy.result())
                print("remote_accuracy: ", remote_accuracy.result())
                accuracy.reset_states()
                local_accuracy.reset_states()
                remote_accuracy.reset_states()
                break
            step(x, y, training=True)
            if training_step % 200 == 0:
                with writer.as_default():
                    c_loss, acc = cls_loss.result(), accuracy.result()

                    tf.summary.scalar('train/accuracy', acc, training_step)
                    tf.summary.scalar('train/skewness', skewness.result(), training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    accuracy.reset_states()
                    local_accuracy.reset_states()
                    remote_accuracy.reset_states()
                    skewness.reset_states()
        
        
        cls_loss.reset_states()
        accuracy.reset_states()
        local_accuracy.reset_states()
        remote_accuracy.reset_states()
        skewness.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/accuracy', accuracy.result(), step=training_step)
            tf.summary.scalar('test/local_accuracy', local_accuracy.result(), step=training_step)
            tf.summary.scalar('test/remote_accuracy', remote_accuracy.result(), step=training_step)
            tf.summary.scalar('test/skewness', skewness.result(), training_step)
            print("=================================")
            print("accuracy: ", accuracy.result())
            print("local_accuracy: ", local_accuracy.result())
            print("remote_accuracy: ", remote_accuracy.result())
            print("skewness: ", skewness.result())
            print("=================================")
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                
            cls_loss.reset_states()
            accuracy.reset_states()
            skewness.reset_states()