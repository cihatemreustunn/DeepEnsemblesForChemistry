��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
�
 Adam/base_model_2/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/base_model_2/dense_8/bias/v
�
4Adam/base_model_2/dense_8/bias/v/Read/ReadVariableOpReadVariableOp Adam/base_model_2/dense_8/bias/v*
_output_shapes
:	*
dtype0
�
"Adam/base_model_2/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	*3
shared_name$"Adam/base_model_2/dense_8/kernel/v
�
6Adam/base_model_2/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/base_model_2/dense_8/kernel/v*
_output_shapes

:@	*
dtype0
�
 Adam/base_model_2/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/base_model_2/dense_7/bias/v
�
4Adam/base_model_2/dense_7/bias/v/Read/ReadVariableOpReadVariableOp Adam/base_model_2/dense_7/bias/v*
_output_shapes
:@*
dtype0
�
"Adam/base_model_2/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*3
shared_name$"Adam/base_model_2/dense_7/kernel/v
�
6Adam/base_model_2/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/base_model_2/dense_7/kernel/v*
_output_shapes

:@@*
dtype0
�
 Adam/base_model_2/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/base_model_2/dense_6/bias/v
�
4Adam/base_model_2/dense_6/bias/v/Read/ReadVariableOpReadVariableOp Adam/base_model_2/dense_6/bias/v*
_output_shapes
:@*
dtype0
�
"Adam/base_model_2/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*3
shared_name$"Adam/base_model_2/dense_6/kernel/v
�
6Adam/base_model_2/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/base_model_2/dense_6/kernel/v*
_output_shapes

:	@*
dtype0
�
 Adam/base_model_2/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/base_model_2/dense_8/bias/m
�
4Adam/base_model_2/dense_8/bias/m/Read/ReadVariableOpReadVariableOp Adam/base_model_2/dense_8/bias/m*
_output_shapes
:	*
dtype0
�
"Adam/base_model_2/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	*3
shared_name$"Adam/base_model_2/dense_8/kernel/m
�
6Adam/base_model_2/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/base_model_2/dense_8/kernel/m*
_output_shapes

:@	*
dtype0
�
 Adam/base_model_2/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/base_model_2/dense_7/bias/m
�
4Adam/base_model_2/dense_7/bias/m/Read/ReadVariableOpReadVariableOp Adam/base_model_2/dense_7/bias/m*
_output_shapes
:@*
dtype0
�
"Adam/base_model_2/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*3
shared_name$"Adam/base_model_2/dense_7/kernel/m
�
6Adam/base_model_2/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/base_model_2/dense_7/kernel/m*
_output_shapes

:@@*
dtype0
�
 Adam/base_model_2/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/base_model_2/dense_6/bias/m
�
4Adam/base_model_2/dense_6/bias/m/Read/ReadVariableOpReadVariableOp Adam/base_model_2/dense_6/bias/m*
_output_shapes
:@*
dtype0
�
"Adam/base_model_2/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*3
shared_name$"Adam/base_model_2/dense_6/kernel/m
�
6Adam/base_model_2/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/base_model_2/dense_6/kernel/m*
_output_shapes

:	@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
base_model_2/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_namebase_model_2/dense_8/bias
�
-base_model_2/dense_8/bias/Read/ReadVariableOpReadVariableOpbase_model_2/dense_8/bias*
_output_shapes
:	*
dtype0
�
base_model_2/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	*,
shared_namebase_model_2/dense_8/kernel
�
/base_model_2/dense_8/kernel/Read/ReadVariableOpReadVariableOpbase_model_2/dense_8/kernel*
_output_shapes

:@	*
dtype0
�
base_model_2/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebase_model_2/dense_7/bias
�
-base_model_2/dense_7/bias/Read/ReadVariableOpReadVariableOpbase_model_2/dense_7/bias*
_output_shapes
:@*
dtype0
�
base_model_2/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*,
shared_namebase_model_2/dense_7/kernel
�
/base_model_2/dense_7/kernel/Read/ReadVariableOpReadVariableOpbase_model_2/dense_7/kernel*
_output_shapes

:@@*
dtype0
�
base_model_2/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebase_model_2/dense_6/bias
�
-base_model_2/dense_6/bias/Read/ReadVariableOpReadVariableOpbase_model_2/dense_6/bias*
_output_shapes
:@*
dtype0
�
base_model_2/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	@*,
shared_namebase_model_2/dense_6/kernel
�
/base_model_2/dense_6/kernel/Read/ReadVariableOpReadVariableOpbase_model_2/dense_6/kernel*
_output_shapes

:	@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1base_model_2/dense_6/kernelbase_model_2/dense_6/biasbase_model_2/dense_7/kernelbase_model_2/dense_7/biasbase_model_2/dense_8/kernelbase_model_2/dense_8/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_6296175

NoOpNoOp
�+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2

output_layer
	optimizer

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

kernel
bias*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

kernel
bias*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

kernel
bias*
�
.iter

/beta_1

0beta_2
	1decay
2learning_ratemNmOmPmQmRmSvTvUvVvWvXvY*

3serving_default* 
[U
VARIABLE_VALUEbase_model_2/dense_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbase_model_2/dense_6/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbase_model_2/dense_7/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbase_model_2/dense_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbase_model_2/dense_8/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbase_model_2/dense_8/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*

40*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

:trace_0* 

;trace_0* 

0
1*

0
1*
* 
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

Atrace_0* 

Btrace_0* 

0
1*

0
1*
* 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

Htrace_0* 

Itrace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
J	variables
K	keras_api
	Ltotal
	Mcount*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

L0
M1*

J	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_2/dense_6/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/base_model_2/dense_6/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_2/dense_7/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/base_model_2/dense_7/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_2/dense_8/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/base_model_2/dense_8/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_2/dense_6/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/base_model_2/dense_6/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_2/dense_7/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/base_model_2/dense_7/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_2/dense_8/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/base_model_2/dense_8/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/base_model_2/dense_6/kernel/Read/ReadVariableOp-base_model_2/dense_6/bias/Read/ReadVariableOp/base_model_2/dense_7/kernel/Read/ReadVariableOp-base_model_2/dense_7/bias/Read/ReadVariableOp/base_model_2/dense_8/kernel/Read/ReadVariableOp-base_model_2/dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp6Adam/base_model_2/dense_6/kernel/m/Read/ReadVariableOp4Adam/base_model_2/dense_6/bias/m/Read/ReadVariableOp6Adam/base_model_2/dense_7/kernel/m/Read/ReadVariableOp4Adam/base_model_2/dense_7/bias/m/Read/ReadVariableOp6Adam/base_model_2/dense_8/kernel/m/Read/ReadVariableOp4Adam/base_model_2/dense_8/bias/m/Read/ReadVariableOp6Adam/base_model_2/dense_6/kernel/v/Read/ReadVariableOp4Adam/base_model_2/dense_6/bias/v/Read/ReadVariableOp6Adam/base_model_2/dense_7/kernel/v/Read/ReadVariableOp4Adam/base_model_2/dense_7/bias/v/Read/ReadVariableOp6Adam/base_model_2/dense_8/kernel/v/Read/ReadVariableOp4Adam/base_model_2/dense_8/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_6296373
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebase_model_2/dense_6/kernelbase_model_2/dense_6/biasbase_model_2/dense_7/kernelbase_model_2/dense_7/biasbase_model_2/dense_8/kernelbase_model_2/dense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount"Adam/base_model_2/dense_6/kernel/m Adam/base_model_2/dense_6/bias/m"Adam/base_model_2/dense_7/kernel/m Adam/base_model_2/dense_7/bias/m"Adam/base_model_2/dense_8/kernel/m Adam/base_model_2/dense_8/bias/m"Adam/base_model_2/dense_6/kernel/v Adam/base_model_2/dense_6/bias/v"Adam/base_model_2/dense_7/kernel/v Adam/base_model_2/dense_7/bias/v"Adam/base_model_2/dense_8/kernel/v Adam/base_model_2/dense_8/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_6296458��
�
�
%__inference_signature_wrapper_6296175
input_1
unknown:	@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@	
	unknown_4:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_6296011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1
�

�
D__inference_dense_7_layer_call_and_return_conditional_losses_6296256

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_dense_6_layer_call_fn_6296225

inputs
unknown:	@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_6296029o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
D__inference_dense_6_layer_call_and_return_conditional_losses_6296029

inputs0
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�f
�
#__inference__traced_restore_6296458
file_prefix>
,assignvariableop_base_model_2_dense_6_kernel:	@:
,assignvariableop_1_base_model_2_dense_6_bias:@@
.assignvariableop_2_base_model_2_dense_7_kernel:@@:
,assignvariableop_3_base_model_2_dense_7_bias:@@
.assignvariableop_4_base_model_2_dense_8_kernel:@	:
,assignvariableop_5_base_model_2_dense_8_bias:	&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: H
6assignvariableop_13_adam_base_model_2_dense_6_kernel_m:	@B
4assignvariableop_14_adam_base_model_2_dense_6_bias_m:@H
6assignvariableop_15_adam_base_model_2_dense_7_kernel_m:@@B
4assignvariableop_16_adam_base_model_2_dense_7_bias_m:@H
6assignvariableop_17_adam_base_model_2_dense_8_kernel_m:@	B
4assignvariableop_18_adam_base_model_2_dense_8_bias_m:	H
6assignvariableop_19_adam_base_model_2_dense_6_kernel_v:	@B
4assignvariableop_20_adam_base_model_2_dense_6_bias_v:@H
6assignvariableop_21_adam_base_model_2_dense_7_kernel_v:@@B
4assignvariableop_22_adam_base_model_2_dense_7_bias_v:@H
6assignvariableop_23_adam_base_model_2_dense_8_kernel_v:@	B
4assignvariableop_24_adam_base_model_2_dense_8_bias_v:	
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp,assignvariableop_base_model_2_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp,assignvariableop_1_base_model_2_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_base_model_2_dense_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_base_model_2_dense_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_base_model_2_dense_8_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_base_model_2_dense_8_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp6assignvariableop_13_adam_base_model_2_dense_6_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adam_base_model_2_dense_6_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp6assignvariableop_15_adam_base_model_2_dense_7_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_base_model_2_dense_7_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_base_model_2_dense_8_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_base_model_2_dense_8_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_base_model_2_dense_6_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_base_model_2_dense_6_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_base_model_2_dense_7_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_base_model_2_dense_7_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_base_model_2_dense_8_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_base_model_2_dense_8_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�!
�
"__inference__wrapped_model_6296011
input_1E
3base_model_2_dense_6_matmul_readvariableop_resource:	@B
4base_model_2_dense_6_biasadd_readvariableop_resource:@E
3base_model_2_dense_7_matmul_readvariableop_resource:@@B
4base_model_2_dense_7_biasadd_readvariableop_resource:@E
3base_model_2_dense_8_matmul_readvariableop_resource:@	B
4base_model_2_dense_8_biasadd_readvariableop_resource:	
identity��+base_model_2/dense_6/BiasAdd/ReadVariableOp�*base_model_2/dense_6/MatMul/ReadVariableOp�+base_model_2/dense_7/BiasAdd/ReadVariableOp�*base_model_2/dense_7/MatMul/ReadVariableOp�+base_model_2/dense_8/BiasAdd/ReadVariableOp�*base_model_2/dense_8/MatMul/ReadVariableOp�
*base_model_2/dense_6/MatMul/ReadVariableOpReadVariableOp3base_model_2_dense_6_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype0�
base_model_2/dense_6/MatMulMatMulinput_12base_model_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+base_model_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4base_model_2_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
base_model_2/dense_6/BiasAddBiasAdd%base_model_2/dense_6/MatMul:product:03base_model_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
base_model_2/dense_6/ReluRelu%base_model_2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*base_model_2/dense_7/MatMul/ReadVariableOpReadVariableOp3base_model_2_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
base_model_2/dense_7/MatMulMatMul'base_model_2/dense_6/Relu:activations:02base_model_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+base_model_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4base_model_2_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
base_model_2/dense_7/BiasAddBiasAdd%base_model_2/dense_7/MatMul:product:03base_model_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
base_model_2/dense_7/ReluRelu%base_model_2/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*base_model_2/dense_8/MatMul/ReadVariableOpReadVariableOp3base_model_2_dense_8_matmul_readvariableop_resource*
_output_shapes

:@	*
dtype0�
base_model_2/dense_8/MatMulMatMul'base_model_2/dense_7/Relu:activations:02base_model_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
+base_model_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4base_model_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
base_model_2/dense_8/BiasAddBiasAdd%base_model_2/dense_8/MatMul:product:03base_model_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	t
IdentityIdentity%base_model_2/dense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp,^base_model_2/dense_6/BiasAdd/ReadVariableOp+^base_model_2/dense_6/MatMul/ReadVariableOp,^base_model_2/dense_7/BiasAdd/ReadVariableOp+^base_model_2/dense_7/MatMul/ReadVariableOp,^base_model_2/dense_8/BiasAdd/ReadVariableOp+^base_model_2/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : : : 2Z
+base_model_2/dense_6/BiasAdd/ReadVariableOp+base_model_2/dense_6/BiasAdd/ReadVariableOp2X
*base_model_2/dense_6/MatMul/ReadVariableOp*base_model_2/dense_6/MatMul/ReadVariableOp2Z
+base_model_2/dense_7/BiasAdd/ReadVariableOp+base_model_2/dense_7/BiasAdd/ReadVariableOp2X
*base_model_2/dense_7/MatMul/ReadVariableOp*base_model_2/dense_7/MatMul/ReadVariableOp2Z
+base_model_2/dense_8/BiasAdd/ReadVariableOp+base_model_2/dense_8/BiasAdd/ReadVariableOp2X
*base_model_2/dense_8/MatMul/ReadVariableOp*base_model_2/dense_8/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1
�
�
)__inference_dense_8_layer_call_fn_6296265

inputs
unknown:@	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_6296062o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_dense_7_layer_call_fn_6296245

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_6296046o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
.__inference_base_model_2_layer_call_fn_6296084
input_1
unknown:	@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@	
	unknown_4:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1
�

�
D__inference_dense_6_layer_call_and_return_conditional_losses_6296236

inputs0
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
D__inference_dense_8_layer_call_and_return_conditional_losses_6296062

inputs0
matmul_readvariableop_resource:@	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
.__inference_base_model_2_layer_call_fn_6296192

inputs
unknown:	@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@	
	unknown_4:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
D__inference_dense_7_layer_call_and_return_conditional_losses_6296046

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296216

inputs8
&dense_6_matmul_readvariableop_resource:	@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@	5
'dense_8_biasadd_readvariableop_resource:	
identity��dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:	@*
dtype0y
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@	*
dtype0�
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	g
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�:
�
 __inference__traced_save_6296373
file_prefix:
6savev2_base_model_2_dense_6_kernel_read_readvariableop8
4savev2_base_model_2_dense_6_bias_read_readvariableop:
6savev2_base_model_2_dense_7_kernel_read_readvariableop8
4savev2_base_model_2_dense_7_bias_read_readvariableop:
6savev2_base_model_2_dense_8_kernel_read_readvariableop8
4savev2_base_model_2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopA
=savev2_adam_base_model_2_dense_6_kernel_m_read_readvariableop?
;savev2_adam_base_model_2_dense_6_bias_m_read_readvariableopA
=savev2_adam_base_model_2_dense_7_kernel_m_read_readvariableop?
;savev2_adam_base_model_2_dense_7_bias_m_read_readvariableopA
=savev2_adam_base_model_2_dense_8_kernel_m_read_readvariableop?
;savev2_adam_base_model_2_dense_8_bias_m_read_readvariableopA
=savev2_adam_base_model_2_dense_6_kernel_v_read_readvariableop?
;savev2_adam_base_model_2_dense_6_bias_v_read_readvariableopA
=savev2_adam_base_model_2_dense_7_kernel_v_read_readvariableop?
;savev2_adam_base_model_2_dense_7_bias_v_read_readvariableopA
=savev2_adam_base_model_2_dense_8_kernel_v_read_readvariableop?
;savev2_adam_base_model_2_dense_8_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_base_model_2_dense_6_kernel_read_readvariableop4savev2_base_model_2_dense_6_bias_read_readvariableop6savev2_base_model_2_dense_7_kernel_read_readvariableop4savev2_base_model_2_dense_7_bias_read_readvariableop6savev2_base_model_2_dense_8_kernel_read_readvariableop4savev2_base_model_2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop=savev2_adam_base_model_2_dense_6_kernel_m_read_readvariableop;savev2_adam_base_model_2_dense_6_bias_m_read_readvariableop=savev2_adam_base_model_2_dense_7_kernel_m_read_readvariableop;savev2_adam_base_model_2_dense_7_bias_m_read_readvariableop=savev2_adam_base_model_2_dense_8_kernel_m_read_readvariableop;savev2_adam_base_model_2_dense_8_bias_m_read_readvariableop=savev2_adam_base_model_2_dense_6_kernel_v_read_readvariableop;savev2_adam_base_model_2_dense_6_bias_v_read_readvariableop=savev2_adam_base_model_2_dense_7_kernel_v_read_readvariableop;savev2_adam_base_model_2_dense_7_bias_v_read_readvariableop=savev2_adam_base_model_2_dense_8_kernel_v_read_readvariableop;savev2_adam_base_model_2_dense_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	@:@:@@:@:@	:	: : : : : : : :	@:@:@@:@:@	:	:	@:@:@@:@:@	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@	: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@	: 

_output_shapes
:	:$ 

_output_shapes

:	@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@	: 

_output_shapes
:	:

_output_shapes
: 
�	
�
D__inference_dense_8_layer_call_and_return_conditional_losses_6296275

inputs0
matmul_readvariableop_resource:@	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296069

inputs!
dense_6_6296030:	@
dense_6_6296032:@!
dense_7_6296047:@@
dense_7_6296049:@!
dense_8_6296063:@	
dense_8_6296065:	
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_6296030dense_6_6296032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_6296029�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_6296047dense_7_6296049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_6296046�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_6296063dense_8_6296065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_6296062w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296150
input_1!
dense_6_6296134:	@
dense_6_6296136:@!
dense_7_6296139:@@
dense_7_6296141:@!
dense_8_6296144:@	
dense_8_6296146:	
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6_6296134dense_6_6296136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_6296029�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_6296139dense_7_6296141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_6296046�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_6296144dense_8_6296146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_6296062w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������	<
output_10
StatefulPartitionedCall:0���������	tensorflow/serving/predict:�Z
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2

output_layer
	optimizer

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
.__inference_base_model_2_layer_call_fn_6296084
.__inference_base_model_2_layer_call_fn_6296192�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296216
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296150�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�B�
"__inference__wrapped_model_6296011input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
.iter

/beta_1

0beta_2
	1decay
2learning_ratemNmOmPmQmRmSvTvUvVvWvXvY"
	optimizer
,
3serving_default"
signature_map
-:+	@2base_model_2/dense_6/kernel
':%@2base_model_2/dense_6/bias
-:+@@2base_model_2/dense_7/kernel
':%@2base_model_2/dense_7/bias
-:+@	2base_model_2/dense_8/kernel
':%	2base_model_2/dense_8/bias
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_base_model_2_layer_call_fn_6296084input_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_base_model_2_layer_call_fn_6296192inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296216inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296150input_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
:trace_02�
)__inference_dense_6_layer_call_fn_6296225�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z:trace_0
�
;trace_02�
D__inference_dense_6_layer_call_and_return_conditional_losses_6296236�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z;trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
Atrace_02�
)__inference_dense_7_layer_call_fn_6296245�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zAtrace_0
�
Btrace_02�
D__inference_dense_7_layer_call_and_return_conditional_losses_6296256�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zBtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
Htrace_02�
)__inference_dense_8_layer_call_fn_6296265�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
�
Itrace_02�
D__inference_dense_8_layer_call_and_return_conditional_losses_6296275�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zItrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
%__inference_signature_wrapper_6296175input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
J	variables
K	keras_api
	Ltotal
	Mcount"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_6_layer_call_fn_6296225inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_6_layer_call_and_return_conditional_losses_6296236inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_7_layer_call_fn_6296245inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_7_layer_call_and_return_conditional_losses_6296256inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_8_layer_call_fn_6296265inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_8_layer_call_and_return_conditional_losses_6296275inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
L0
M1"
trackable_list_wrapper
-
J	variables"
_generic_user_object
:  (2total
:  (2count
2:0	@2"Adam/base_model_2/dense_6/kernel/m
,:*@2 Adam/base_model_2/dense_6/bias/m
2:0@@2"Adam/base_model_2/dense_7/kernel/m
,:*@2 Adam/base_model_2/dense_7/bias/m
2:0@	2"Adam/base_model_2/dense_8/kernel/m
,:*	2 Adam/base_model_2/dense_8/bias/m
2:0	@2"Adam/base_model_2/dense_6/kernel/v
,:*@2 Adam/base_model_2/dense_6/bias/v
2:0@@2"Adam/base_model_2/dense_7/kernel/v
,:*@2 Adam/base_model_2/dense_7/bias/v
2:0@	2"Adam/base_model_2/dense_8/kernel/v
,:*	2 Adam/base_model_2/dense_8/bias/v�
"__inference__wrapped_model_6296011o0�-
&�#
!�
input_1���������	
� "3�0
.
output_1"�
output_1���������	�
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296150a0�-
&�#
!�
input_1���������	
� "%�"
�
0���������	
� �
I__inference_base_model_2_layer_call_and_return_conditional_losses_6296216`/�,
%�"
 �
inputs���������	
� "%�"
�
0���������	
� �
.__inference_base_model_2_layer_call_fn_6296084T0�-
&�#
!�
input_1���������	
� "����������	�
.__inference_base_model_2_layer_call_fn_6296192S/�,
%�"
 �
inputs���������	
� "����������	�
D__inference_dense_6_layer_call_and_return_conditional_losses_6296236\/�,
%�"
 �
inputs���������	
� "%�"
�
0���������@
� |
)__inference_dense_6_layer_call_fn_6296225O/�,
%�"
 �
inputs���������	
� "����������@�
D__inference_dense_7_layer_call_and_return_conditional_losses_6296256\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_dense_7_layer_call_fn_6296245O/�,
%�"
 �
inputs���������@
� "����������@�
D__inference_dense_8_layer_call_and_return_conditional_losses_6296275\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������	
� |
)__inference_dense_8_layer_call_fn_6296265O/�,
%�"
 �
inputs���������@
� "����������	�
%__inference_signature_wrapper_6296175z;�8
� 
1�.
,
input_1!�
input_1���������	"3�0
.
output_1"�
output_1���������	