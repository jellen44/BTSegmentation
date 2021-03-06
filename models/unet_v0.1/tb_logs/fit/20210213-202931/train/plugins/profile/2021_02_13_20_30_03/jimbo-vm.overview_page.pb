?	?0???2?@?0???2?@!?0???2?@	0k$????0k$????!0k$????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?0???2?@????@16?!}??@Aq8??9@??IG???e??YF?@12??*	a??"?mf@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?4LkӰ?!u?7?PB@)ŪA?۽??1??=?I?@:Preprocessing2U
Iterator::Model::ParallelMapV2us??=A??!?????3@)us??=A??1?????3@:Preprocessing2F
Iterator::Model???w?̰?!e??2IB@)?ôo?1??i??0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??`?>??!?w2[w?%@)??`?>??1?w2[w?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateU?3?Y??!f?!5@)????9???1?T?*?Y$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???죃?!?ĻO?`@)???죃?1?ĻO?`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipx?캷"??!??ͶO@)eQ?E??1n}?=?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?- ????!P?NZ?6@)}(Ff?1mc?t?>??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no90k$????I???Z8???Q9??C??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????@????@!????@      ??!       "	6?!}??@6?!}??@!6?!}??@*      ??!       2	q8??9@??q8??9@??!q8??9@??:	G???e??G???e??!G???e??B      ??!       J	F?@12??F?@12??!F?@12??R      ??!       Z	F?@12??F?@12??!F?@12??b      ??!       JGPUY0k$????b q???Z8???y9??C??X@?"-
IteratorGetNext/_3_Sendg????z??!g????z??"e
9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterT?6jA֨?!??????0"-
IteratorGetNext/_1_Send??	?Ŧ?!)
??x$??"e
9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??M???!d?(h\???0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?/?C??!???
??0"e
9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterJ?Z?R ??!n?!?j??0"c
8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputConv2DBackpropInput??٢????!??*Kk@??0"6
model/conv2d_16/Relu_FusedConv2D&?hĀ??!???w???"c
8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputConv2DBackpropInput?x?rF???!????[???0"e
9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterC??󣸓?!??8???0Q      Y@Y+?Z??@a??U*@?W@q& ??9??y6?hV}A?"?	
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 