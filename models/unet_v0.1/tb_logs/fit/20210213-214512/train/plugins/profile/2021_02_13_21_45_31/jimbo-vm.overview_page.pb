?	?????@?????@!?????@	??Pʯ????Pʯ??!??Pʯ??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?????@(??G @1i;???؁@A\:?<c??I?
????Y?i?????*?~j?t'h@)       =2U
Iterator::Model::ParallelMapV27 !???!?8%q$?<@)7 !???1?8%q$?<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???e???!????M9@)?{???1?{???4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateH2?w???!?s$u?e:@)?3?c?=??1????m1@:Preprocessing2F
Iterator::Model?t???l??!E!?O??E@)N?g\W??1?\p?,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??M(D???!??~+?!@)??M(D???1??~+?!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorD?K?KƁ?!??C?@)D?K?KƁ?1??C?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB??=?
??!??f?XL@)??w?'-|?1?D???z@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?C5%Y???!6i0?;@)f??
?f?1?%?>O???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??Pʯ??I??y?m??Q????9?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	(??G @(??G @!(??G @      ??!       "	i;???؁@i;???؁@!i;???؁@*      ??!       2	\:?<c??\:?<c??!\:?<c??:	?
?????
????!?
????B      ??!       J	?i??????i?????!?i?????R      ??!       Z	?i??????i?????!?i?????b      ??!       JGPUY??Pʯ??b q??y?m??y????9?X@?"-
IteratorGetNext/_3_Send[`??b???![`??b???"e
9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterI7F???!??}/u??0"-
IteratorGetNext/_1_Send6???j??!?{?s????"e
9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ń3?
??!t?Q?D??0"e
9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???z???!?p?	Br??0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???/?L??!?/??????0"c
8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputConv2DBackpropInput?J??????!?l?TK???0"6
model/conv2d_16/Relu_FusedConv2D????dΕ?!??˟1V??"c
8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputConv2DBackpropInput]w?=??!???S^???0"e
9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???z????!?bO[f???0Q      Y@Y4H?4H?@a|˷|˷W@q??W????y?^k>?G?"?	
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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