?	?_??~c@?_??~c@!?_??~c@	?C?_????C?_???!?C?_???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?_??~c@????9?@1"? ??b@A?x?Z??I.?R\U???Y??$xC??*	???(\?[@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatDj??4ӝ?!?k??U:@)E?Ɵ?l??1	}??^5@:Preprocessing2U
Iterator::Model::ParallelMapV2毐?2??!c?%? ,5@)毐?2??1c?%? ,5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???oaݠ?!J??Z??=@)?fI-??1j??yN?1@:Preprocessing2F
Iterator::ModelD??)X???!}f?ҘFB@)|?i?????1-???a?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceB?L????!??p?'?'@)B?L????1??p?'?'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip=??? !??!??r-g?O@)m?i?*?y?1?WR<?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??N?0?u?!_?]Q??@)??N?0?u?1_?]Q??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?@?"??!t{~i??@)??
?c?1\?P9D@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?C?_???I??0??@Q?V?BX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????9?@????9?@!????9?@      ??!       "	"? ??b@"? ??b@!"? ??b@*      ??!       2	?x?Z???x?Z??!?x?Z??:	.?R\U???.?R\U???!.?R\U???B      ??!       J	??$xC????$xC??!??$xC??R      ??!       Z	??$xC????$xC??!??$xC??b      ??!       JGPUY?C?_???b q??0??@y?V?BX@?"-
IteratorGetNext/_3_Send?O?rS??!?O?rS??"e
9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?}?)??!?6)?>F??0"-
IteratorGetNext/_1_Send6R.(???!?(?V)H??"e
9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??U????!???K???0"e
9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter@w˕S??!|?th???0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?JOb?,??!ڗ?4PN??0"c
8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputConv2DBackpropInput?`tQI??!?Q?1???0"6
model/conv2d_16/Relu_FusedConv2D??X?2s??!???^?R??"e
9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter\ߕ+?&??!???_???0"d
8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterX?<[ ??!԰x?d???0Q      Y@YY?7?"?@a?????W@q?T ????y?VT峡`?"?	
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