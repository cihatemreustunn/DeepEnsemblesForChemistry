��
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
 �"serve*2.10.02unknown8��
�
"Adam/base_model_15/dense_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/base_model_15/dense_63/bias/v
�
6Adam/base_model_15/dense_63/bias/v/Read/ReadVariableOpReadVariableOp"Adam/base_model_15/dense_63/bias/v*
_output_shapes
:*
dtype0
�
$Adam/base_model_15/dense_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/base_model_15/dense_63/kernel/v
�
8Adam/base_model_15/dense_63/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/base_model_15/dense_63/kernel/v*
_output_shapes

:*
dtype0
�
"Adam/base_model_15/dense_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/base_model_15/dense_62/bias/v
�
6Adam/base_model_15/dense_62/bias/v/Read/ReadVariableOpReadVariableOp"Adam/base_model_15/dense_62/bias/v*
_output_shapes
:*
dtype0
�
$Adam/base_model_15/dense_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *5
shared_name&$Adam/base_model_15/dense_62/kernel/v
�
8Adam/base_model_15/dense_62/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/base_model_15/dense_62/kernel/v*
_output_shapes

: *
dtype0
�
"Adam/base_model_15/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/base_model_15/dense_61/bias/v
�
6Adam/base_model_15/dense_61/bias/v/Read/ReadVariableOpReadVariableOp"Adam/base_model_15/dense_61/bias/v*
_output_shapes
: *
dtype0
�
$Adam/base_model_15/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *5
shared_name&$Adam/base_model_15/dense_61/kernel/v
�
8Adam/base_model_15/dense_61/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/base_model_15/dense_61/kernel/v*
_output_shapes

: *
dtype0
�
"Adam/base_model_15/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/base_model_15/dense_60/bias/v
�
6Adam/base_model_15/dense_60/bias/v/Read/ReadVariableOpReadVariableOp"Adam/base_model_15/dense_60/bias/v*
_output_shapes
:*
dtype0
�
$Adam/base_model_15/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*5
shared_name&$Adam/base_model_15/dense_60/kernel/v
�
8Adam/base_model_15/dense_60/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/base_model_15/dense_60/kernel/v*
_output_shapes

:	*
dtype0
�
"Adam/base_model_15/dense_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/base_model_15/dense_63/bias/m
�
6Adam/base_model_15/dense_63/bias/m/Read/ReadVariableOpReadVariableOp"Adam/base_model_15/dense_63/bias/m*
_output_shapes
:*
dtype0
�
$Adam/base_model_15/dense_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/base_model_15/dense_63/kernel/m
�
8Adam/base_model_15/dense_63/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/base_model_15/dense_63/kernel/m*
_output_shapes

:*
dtype0
�
"Adam/base_model_15/dense_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/base_model_15/dense_62/bias/m
�
6Adam/base_model_15/dense_62/bias/m/Read/ReadVariableOpReadVariableOp"Adam/base_model_15/dense_62/bias/m*
_output_shapes
:*
dtype0
�
$Adam/base_model_15/dense_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *5
shared_name&$Adam/base_model_15/dense_62/kernel/m
�
8Adam/base_model_15/dense_62/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/base_model_15/dense_62/kernel/m*
_output_shapes

: *
dtype0
�
"Adam/base_model_15/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/base_model_15/dense_61/bias/m
�
6Adam/base_model_15/dense_61/bias/m/Read/ReadVariableOpReadVariableOp"Adam/base_model_15/dense_61/bias/m*
_output_shapes
: *
dtype0
�
$Adam/base_model_15/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *5
shared_name&$Adam/base_model_15/dense_61/kernel/m
�
8Adam/base_model_15/dense_61/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/base_model_15/dense_61/kernel/m*
_output_shapes

: *
dtype0
�
"Adam/base_model_15/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/base_model_15/dense_60/bias/m
�
6Adam/base_model_15/dense_60/bias/m/Read/ReadVariableOpReadVariableOp"Adam/base_model_15/dense_60/bias/m*
_output_shapes
:*
dtype0
�
$Adam/base_model_15/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*5
shared_name&$Adam/base_model_15/dense_60/kernel/m
�
8Adam/base_model_15/dense_60/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/base_model_15/dense_60/kernel/m*
_output_shapes

:	*
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
base_model_15/dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebase_model_15/dense_63/bias
�
/base_model_15/dense_63/bias/Read/ReadVariableOpReadVariableOpbase_model_15/dense_63/bias*
_output_shapes
:*
dtype0
�
base_model_15/dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namebase_model_15/dense_63/kernel
�
1base_model_15/dense_63/kernel/Read/ReadVariableOpReadVariableOpbase_model_15/dense_63/kernel*
_output_shapes

:*
dtype0
�
base_model_15/dense_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebase_model_15/dense_62/bias
�
/base_model_15/dense_62/bias/Read/ReadVariableOpReadVariableOpbase_model_15/dense_62/bias*
_output_shapes
:*
dtype0
�
base_model_15/dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_namebase_model_15/dense_62/kernel
�
1base_model_15/dense_62/kernel/Read/ReadVariableOpReadVariableOpbase_model_15/dense_62/kernel*
_output_shapes

: *
dtype0
�
base_model_15/dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebase_model_15/dense_61/bias
�
/base_model_15/dense_61/bias/Read/ReadVariableOpReadVariableOpbase_model_15/dense_61/bias*
_output_shapes
: *
dtype0
�
base_model_15/dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_namebase_model_15/dense_61/kernel
�
1base_model_15/dense_61/kernel/Read/ReadVariableOpReadVariableOpbase_model_15/dense_61/kernel*
_output_shapes

: *
dtype0
�
base_model_15/dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebase_model_15/dense_60/bias
�
/base_model_15/dense_60/bias/Read/ReadVariableOpReadVariableOpbase_model_15/dense_60/bias*
_output_shapes
:*
dtype0
�
base_model_15/dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*.
shared_namebase_model_15/dense_60/kernel
�
1base_model_15/dense_60/kernel/Read/ReadVariableOpReadVariableOpbase_model_15/dense_60/kernel*
_output_shapes

:	*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1base_model_15/dense_60/kernelbase_model_15/dense_60/biasbase_model_15/dense_61/kernelbase_model_15/dense_61/biasbase_model_15/dense_62/kernelbase_model_15/dense_62/biasbase_model_15/dense_63/kernelbase_model_15/dense_63/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_11155493

NoOpNoOp
�5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�5
value�5B�5 B�5
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


dense3
output_layer
	optimizer

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias*
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias*
�
7iter

8beta_1

9beta_2
	:decay
;learning_ratem^m_m`mambmcmdmevfvgvhvivjvkvlvm*

<serving_default* 
]W
VARIABLE_VALUEbase_model_15/dense_60/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbase_model_15/dense_60/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbase_model_15/dense_61/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbase_model_15/dense_61/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbase_model_15/dense_62/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbase_model_15/dense_62/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbase_model_15/dense_63/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbase_model_15/dense_63/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*

=0*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 

0
1*

0
1*
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 

0
1*

0
1*
* 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 

0
1*

0
1*
* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
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
Z	variables
[	keras_api
	\total
	]count*
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
* 
* 
* 
* 
* 
* 
* 

\0
]1*

Z	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/base_model_15/dense_60/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_15/dense_60/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/base_model_15/dense_61/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_15/dense_61/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/base_model_15/dense_62/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_15/dense_62/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/base_model_15/dense_63/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_15/dense_63/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/base_model_15/dense_60/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_15/dense_60/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/base_model_15/dense_61/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_15/dense_61/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/base_model_15/dense_62/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_15/dense_62/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE$Adam/base_model_15/dense_63/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/base_model_15/dense_63/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1base_model_15/dense_60/kernel/Read/ReadVariableOp/base_model_15/dense_60/bias/Read/ReadVariableOp1base_model_15/dense_61/kernel/Read/ReadVariableOp/base_model_15/dense_61/bias/Read/ReadVariableOp1base_model_15/dense_62/kernel/Read/ReadVariableOp/base_model_15/dense_62/bias/Read/ReadVariableOp1base_model_15/dense_63/kernel/Read/ReadVariableOp/base_model_15/dense_63/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp8Adam/base_model_15/dense_60/kernel/m/Read/ReadVariableOp6Adam/base_model_15/dense_60/bias/m/Read/ReadVariableOp8Adam/base_model_15/dense_61/kernel/m/Read/ReadVariableOp6Adam/base_model_15/dense_61/bias/m/Read/ReadVariableOp8Adam/base_model_15/dense_62/kernel/m/Read/ReadVariableOp6Adam/base_model_15/dense_62/bias/m/Read/ReadVariableOp8Adam/base_model_15/dense_63/kernel/m/Read/ReadVariableOp6Adam/base_model_15/dense_63/bias/m/Read/ReadVariableOp8Adam/base_model_15/dense_60/kernel/v/Read/ReadVariableOp6Adam/base_model_15/dense_60/bias/v/Read/ReadVariableOp8Adam/base_model_15/dense_61/kernel/v/Read/ReadVariableOp6Adam/base_model_15/dense_61/bias/v/Read/ReadVariableOp8Adam/base_model_15/dense_62/kernel/v/Read/ReadVariableOp6Adam/base_model_15/dense_62/bias/v/Read/ReadVariableOp8Adam/base_model_15/dense_63/kernel/v/Read/ReadVariableOp6Adam/base_model_15/dense_63/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
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
GPU 2J 8� **
f%R#
!__inference__traced_save_11155740
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebase_model_15/dense_60/kernelbase_model_15/dense_60/biasbase_model_15/dense_61/kernelbase_model_15/dense_61/biasbase_model_15/dense_62/kernelbase_model_15/dense_62/biasbase_model_15/dense_63/kernelbase_model_15/dense_63/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount$Adam/base_model_15/dense_60/kernel/m"Adam/base_model_15/dense_60/bias/m$Adam/base_model_15/dense_61/kernel/m"Adam/base_model_15/dense_61/bias/m$Adam/base_model_15/dense_62/kernel/m"Adam/base_model_15/dense_62/bias/m$Adam/base_model_15/dense_63/kernel/m"Adam/base_model_15/dense_63/bias/m$Adam/base_model_15/dense_60/kernel/v"Adam/base_model_15/dense_60/bias/v$Adam/base_model_15/dense_61/kernel/v"Adam/base_model_15/dense_61/bias/v$Adam/base_model_15/dense_62/kernel/v"Adam/base_model_15/dense_62/bias/v$Adam/base_model_15/dense_63/kernel/v"Adam/base_model_15/dense_63/bias/v*+
Tin$
"2 *
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
GPU 2J 8� *-
f(R&
$__inference__traced_restore_11155843��
�

�
F__inference_dense_60_layer_call_and_return_conditional_losses_11155565

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
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
F__inference_dense_63_layer_call_and_return_conditional_losses_11155624

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_62_layer_call_and_return_conditional_losses_11155337

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_62_layer_call_and_return_conditional_losses_11155605

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
F__inference_dense_63_layer_call_and_return_conditional_losses_11155353

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_60_layer_call_and_return_conditional_losses_11155303

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
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
�#
�
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155545

inputs9
'dense_60_matmul_readvariableop_resource:	6
(dense_60_biasadd_readvariableop_resource:9
'dense_61_matmul_readvariableop_resource: 6
(dense_61_biasadd_readvariableop_resource: 9
'dense_62_matmul_readvariableop_resource: 6
(dense_62_biasadd_readvariableop_resource:9
'dense_63_matmul_readvariableop_resource:6
(dense_63_biasadd_readvariableop_resource:
identity��dense_60/BiasAdd/ReadVariableOp�dense_60/MatMul/ReadVariableOp�dense_61/BiasAdd/ReadVariableOp�dense_61/MatMul/ReadVariableOp�dense_62/BiasAdd/ReadVariableOp�dense_62/MatMul/ReadVariableOp�dense_63/BiasAdd/ReadVariableOp�dense_63/MatMul/ReadVariableOp�
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0{
dense_60/MatMulMatMulinputs&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_62/MatMulMatMuldense_61/Relu:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_62/ReluReludense_62/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_63/MatMulMatMuldense_62/Relu:activations:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_63/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
&__inference_signature_wrapper_11155493
input_1
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_11155285o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1
�

�
F__inference_dense_61_layer_call_and_return_conditional_losses_11155585

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_dense_63_layer_call_fn_11155614

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_63_layer_call_and_return_conditional_losses_11155353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�E
�
!__inference__traced_save_11155740
file_prefix<
8savev2_base_model_15_dense_60_kernel_read_readvariableop:
6savev2_base_model_15_dense_60_bias_read_readvariableop<
8savev2_base_model_15_dense_61_kernel_read_readvariableop:
6savev2_base_model_15_dense_61_bias_read_readvariableop<
8savev2_base_model_15_dense_62_kernel_read_readvariableop:
6savev2_base_model_15_dense_62_bias_read_readvariableop<
8savev2_base_model_15_dense_63_kernel_read_readvariableop:
6savev2_base_model_15_dense_63_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopC
?savev2_adam_base_model_15_dense_60_kernel_m_read_readvariableopA
=savev2_adam_base_model_15_dense_60_bias_m_read_readvariableopC
?savev2_adam_base_model_15_dense_61_kernel_m_read_readvariableopA
=savev2_adam_base_model_15_dense_61_bias_m_read_readvariableopC
?savev2_adam_base_model_15_dense_62_kernel_m_read_readvariableopA
=savev2_adam_base_model_15_dense_62_bias_m_read_readvariableopC
?savev2_adam_base_model_15_dense_63_kernel_m_read_readvariableopA
=savev2_adam_base_model_15_dense_63_bias_m_read_readvariableopC
?savev2_adam_base_model_15_dense_60_kernel_v_read_readvariableopA
=savev2_adam_base_model_15_dense_60_bias_v_read_readvariableopC
?savev2_adam_base_model_15_dense_61_kernel_v_read_readvariableopA
=savev2_adam_base_model_15_dense_61_bias_v_read_readvariableopC
?savev2_adam_base_model_15_dense_62_kernel_v_read_readvariableopA
=savev2_adam_base_model_15_dense_62_bias_v_read_readvariableopC
?savev2_adam_base_model_15_dense_63_kernel_v_read_readvariableopA
=savev2_adam_base_model_15_dense_63_bias_v_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_base_model_15_dense_60_kernel_read_readvariableop6savev2_base_model_15_dense_60_bias_read_readvariableop8savev2_base_model_15_dense_61_kernel_read_readvariableop6savev2_base_model_15_dense_61_bias_read_readvariableop8savev2_base_model_15_dense_62_kernel_read_readvariableop6savev2_base_model_15_dense_62_bias_read_readvariableop8savev2_base_model_15_dense_63_kernel_read_readvariableop6savev2_base_model_15_dense_63_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop?savev2_adam_base_model_15_dense_60_kernel_m_read_readvariableop=savev2_adam_base_model_15_dense_60_bias_m_read_readvariableop?savev2_adam_base_model_15_dense_61_kernel_m_read_readvariableop=savev2_adam_base_model_15_dense_61_bias_m_read_readvariableop?savev2_adam_base_model_15_dense_62_kernel_m_read_readvariableop=savev2_adam_base_model_15_dense_62_bias_m_read_readvariableop?savev2_adam_base_model_15_dense_63_kernel_m_read_readvariableop=savev2_adam_base_model_15_dense_63_bias_m_read_readvariableop?savev2_adam_base_model_15_dense_60_kernel_v_read_readvariableop=savev2_adam_base_model_15_dense_60_bias_v_read_readvariableop?savev2_adam_base_model_15_dense_61_kernel_v_read_readvariableop=savev2_adam_base_model_15_dense_61_bias_v_read_readvariableop?savev2_adam_base_model_15_dense_62_kernel_v_read_readvariableop=savev2_adam_base_model_15_dense_62_bias_v_read_readvariableop?savev2_adam_base_model_15_dense_63_kernel_v_read_readvariableop=savev2_adam_base_model_15_dense_63_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	�
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
�: :	:: : : :::: : : : : : : :	:: : : ::::	:: : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	
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
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
�
�
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155360

inputs#
dense_60_11155304:	
dense_60_11155306:#
dense_61_11155321: 
dense_61_11155323: #
dense_62_11155338: 
dense_62_11155340:#
dense_63_11155354:
dense_63_11155356:
identity�� dense_60/StatefulPartitionedCall� dense_61/StatefulPartitionedCall� dense_62/StatefulPartitionedCall� dense_63/StatefulPartitionedCall�
 dense_60/StatefulPartitionedCallStatefulPartitionedCallinputsdense_60_11155304dense_60_11155306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_60_layer_call_and_return_conditional_losses_11155303�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_11155321dense_61_11155323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_11155320�
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_11155338dense_62_11155340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_62_layer_call_and_return_conditional_losses_11155337�
 dense_63/StatefulPartitionedCallStatefulPartitionedCall)dense_62/StatefulPartitionedCall:output:0dense_63_11155354dense_63_11155356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_63_layer_call_and_return_conditional_losses_11155353x
IdentityIdentity)dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
+__inference_dense_60_layer_call_fn_11155554

inputs
unknown:	
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_60_layer_call_and_return_conditional_losses_11155303o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
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
�
�
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155464
input_1#
dense_60_11155443:	
dense_60_11155445:#
dense_61_11155448: 
dense_61_11155450: #
dense_62_11155453: 
dense_62_11155455:#
dense_63_11155458:
dense_63_11155460:
identity�� dense_60/StatefulPartitionedCall� dense_61/StatefulPartitionedCall� dense_62/StatefulPartitionedCall� dense_63/StatefulPartitionedCall�
 dense_60/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_60_11155443dense_60_11155445*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_60_layer_call_and_return_conditional_losses_11155303�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0dense_61_11155448dense_61_11155450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_11155320�
 dense_62/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0dense_62_11155453dense_62_11155455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_62_layer_call_and_return_conditional_losses_11155337�
 dense_63/StatefulPartitionedCallStatefulPartitionedCall)dense_62/StatefulPartitionedCall:output:0dense_63_11155458dense_63_11155460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_63_layer_call_and_return_conditional_losses_11155353x
IdentityIdentity)dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1
�	
�
0__inference_base_model_15_layer_call_fn_11155514

inputs
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
$__inference__traced_restore_11155843
file_prefix@
.assignvariableop_base_model_15_dense_60_kernel:	<
.assignvariableop_1_base_model_15_dense_60_bias:B
0assignvariableop_2_base_model_15_dense_61_kernel: <
.assignvariableop_3_base_model_15_dense_61_bias: B
0assignvariableop_4_base_model_15_dense_62_kernel: <
.assignvariableop_5_base_model_15_dense_62_bias:B
0assignvariableop_6_base_model_15_dense_63_kernel:<
.assignvariableop_7_base_model_15_dense_63_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: J
8assignvariableop_15_adam_base_model_15_dense_60_kernel_m:	D
6assignvariableop_16_adam_base_model_15_dense_60_bias_m:J
8assignvariableop_17_adam_base_model_15_dense_61_kernel_m: D
6assignvariableop_18_adam_base_model_15_dense_61_bias_m: J
8assignvariableop_19_adam_base_model_15_dense_62_kernel_m: D
6assignvariableop_20_adam_base_model_15_dense_62_bias_m:J
8assignvariableop_21_adam_base_model_15_dense_63_kernel_m:D
6assignvariableop_22_adam_base_model_15_dense_63_bias_m:J
8assignvariableop_23_adam_base_model_15_dense_60_kernel_v:	D
6assignvariableop_24_adam_base_model_15_dense_60_bias_v:J
8assignvariableop_25_adam_base_model_15_dense_61_kernel_v: D
6assignvariableop_26_adam_base_model_15_dense_61_bias_v: J
8assignvariableop_27_adam_base_model_15_dense_62_kernel_v: D
6assignvariableop_28_adam_base_model_15_dense_62_bias_v:J
8assignvariableop_29_adam_base_model_15_dense_63_kernel_v:D
6assignvariableop_30_adam_base_model_15_dense_63_bias_v:
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp.assignvariableop_base_model_15_dense_60_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_base_model_15_dense_60_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_base_model_15_dense_61_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_base_model_15_dense_61_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_base_model_15_dense_62_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_base_model_15_dense_62_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp0assignvariableop_6_base_model_15_dense_63_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_base_model_15_dense_63_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp8assignvariableop_15_adam_base_model_15_dense_60_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_base_model_15_dense_60_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_base_model_15_dense_61_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_base_model_15_dense_61_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_base_model_15_dense_62_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_base_model_15_dense_62_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp8assignvariableop_21_adam_base_model_15_dense_63_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_base_model_15_dense_63_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adam_base_model_15_dense_60_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_base_model_15_dense_60_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_base_model_15_dense_61_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_base_model_15_dense_61_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp8assignvariableop_27_adam_base_model_15_dense_62_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_base_model_15_dense_62_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp8assignvariableop_29_adam_base_model_15_dense_63_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_base_model_15_dense_63_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
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
�
�
+__inference_dense_62_layer_call_fn_11155594

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_62_layer_call_and_return_conditional_losses_11155337o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�,
�
#__inference__wrapped_model_11155285
input_1G
5base_model_15_dense_60_matmul_readvariableop_resource:	D
6base_model_15_dense_60_biasadd_readvariableop_resource:G
5base_model_15_dense_61_matmul_readvariableop_resource: D
6base_model_15_dense_61_biasadd_readvariableop_resource: G
5base_model_15_dense_62_matmul_readvariableop_resource: D
6base_model_15_dense_62_biasadd_readvariableop_resource:G
5base_model_15_dense_63_matmul_readvariableop_resource:D
6base_model_15_dense_63_biasadd_readvariableop_resource:
identity��-base_model_15/dense_60/BiasAdd/ReadVariableOp�,base_model_15/dense_60/MatMul/ReadVariableOp�-base_model_15/dense_61/BiasAdd/ReadVariableOp�,base_model_15/dense_61/MatMul/ReadVariableOp�-base_model_15/dense_62/BiasAdd/ReadVariableOp�,base_model_15/dense_62/MatMul/ReadVariableOp�-base_model_15/dense_63/BiasAdd/ReadVariableOp�,base_model_15/dense_63/MatMul/ReadVariableOp�
,base_model_15/dense_60/MatMul/ReadVariableOpReadVariableOp5base_model_15_dense_60_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0�
base_model_15/dense_60/MatMulMatMulinput_14base_model_15/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-base_model_15/dense_60/BiasAdd/ReadVariableOpReadVariableOp6base_model_15_dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
base_model_15/dense_60/BiasAddBiasAdd'base_model_15/dense_60/MatMul:product:05base_model_15/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
base_model_15/dense_60/ReluRelu'base_model_15/dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,base_model_15/dense_61/MatMul/ReadVariableOpReadVariableOp5base_model_15_dense_61_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
base_model_15/dense_61/MatMulMatMul)base_model_15/dense_60/Relu:activations:04base_model_15/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-base_model_15/dense_61/BiasAdd/ReadVariableOpReadVariableOp6base_model_15_dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
base_model_15/dense_61/BiasAddBiasAdd'base_model_15/dense_61/MatMul:product:05base_model_15/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� ~
base_model_15/dense_61/ReluRelu'base_model_15/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
,base_model_15/dense_62/MatMul/ReadVariableOpReadVariableOp5base_model_15_dense_62_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
base_model_15/dense_62/MatMulMatMul)base_model_15/dense_61/Relu:activations:04base_model_15/dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-base_model_15/dense_62/BiasAdd/ReadVariableOpReadVariableOp6base_model_15_dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
base_model_15/dense_62/BiasAddBiasAdd'base_model_15/dense_62/MatMul:product:05base_model_15/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
base_model_15/dense_62/ReluRelu'base_model_15/dense_62/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,base_model_15/dense_63/MatMul/ReadVariableOpReadVariableOp5base_model_15_dense_63_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
base_model_15/dense_63/MatMulMatMul)base_model_15/dense_62/Relu:activations:04base_model_15/dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-base_model_15/dense_63/BiasAdd/ReadVariableOpReadVariableOp6base_model_15_dense_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
base_model_15/dense_63/BiasAddBiasAdd'base_model_15/dense_63/MatMul:product:05base_model_15/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'base_model_15/dense_63/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^base_model_15/dense_60/BiasAdd/ReadVariableOp-^base_model_15/dense_60/MatMul/ReadVariableOp.^base_model_15/dense_61/BiasAdd/ReadVariableOp-^base_model_15/dense_61/MatMul/ReadVariableOp.^base_model_15/dense_62/BiasAdd/ReadVariableOp-^base_model_15/dense_62/MatMul/ReadVariableOp.^base_model_15/dense_63/BiasAdd/ReadVariableOp-^base_model_15/dense_63/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 2^
-base_model_15/dense_60/BiasAdd/ReadVariableOp-base_model_15/dense_60/BiasAdd/ReadVariableOp2\
,base_model_15/dense_60/MatMul/ReadVariableOp,base_model_15/dense_60/MatMul/ReadVariableOp2^
-base_model_15/dense_61/BiasAdd/ReadVariableOp-base_model_15/dense_61/BiasAdd/ReadVariableOp2\
,base_model_15/dense_61/MatMul/ReadVariableOp,base_model_15/dense_61/MatMul/ReadVariableOp2^
-base_model_15/dense_62/BiasAdd/ReadVariableOp-base_model_15/dense_62/BiasAdd/ReadVariableOp2\
,base_model_15/dense_62/MatMul/ReadVariableOp,base_model_15/dense_62/MatMul/ReadVariableOp2^
-base_model_15/dense_63/BiasAdd/ReadVariableOp-base_model_15/dense_63/BiasAdd/ReadVariableOp2\
,base_model_15/dense_63/MatMul/ReadVariableOp,base_model_15/dense_63/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1
�
�
+__inference_dense_61_layer_call_fn_11155574

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_11155320o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
0__inference_base_model_15_layer_call_fn_11155379
input_1
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1
�

�
F__inference_dense_61_layer_call_and_return_conditional_losses_11155320

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	L
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:�o
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


dense3
output_layer
	optimizer

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
0__inference_base_model_15_layer_call_fn_11155379
0__inference_base_model_15_layer_call_fn_11155514�
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
 ztrace_0ztrace_1
�
trace_0
trace_12�
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155545
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155464�
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
 ztrace_0ztrace_1
�B�
#__inference__wrapped_model_11155285input_1"�
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
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
7iter

8beta_1

9beta_2
	:decay
;learning_ratem^m_m`mambmcmdmevfvgvhvivjvkvlvm"
	optimizer
,
<serving_default"
signature_map
/:-	2base_model_15/dense_60/kernel
):'2base_model_15/dense_60/bias
/:- 2base_model_15/dense_61/kernel
):' 2base_model_15/dense_61/bias
/:- 2base_model_15/dense_62/kernel
):'2base_model_15/dense_62/bias
/:-2base_model_15/dense_63/kernel
):'2base_model_15/dense_63/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_base_model_15_layer_call_fn_11155379input_1"�
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
0__inference_base_model_15_layer_call_fn_11155514inputs"�
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
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155545inputs"�
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
�B�
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155464input_1"�
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_02�
+__inference_dense_60_layer_call_fn_11155554�
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
 zCtrace_0
�
Dtrace_02�
F__inference_dense_60_layer_call_and_return_conditional_losses_11155565�
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
 zDtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_02�
+__inference_dense_61_layer_call_fn_11155574�
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
 zJtrace_0
�
Ktrace_02�
F__inference_dense_61_layer_call_and_return_conditional_losses_11155585�
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
 zKtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
Qtrace_02�
+__inference_dense_62_layer_call_fn_11155594�
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
 zQtrace_0
�
Rtrace_02�
F__inference_dense_62_layer_call_and_return_conditional_losses_11155605�
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
 zRtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_02�
+__inference_dense_63_layer_call_fn_11155614�
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
 zXtrace_0
�
Ytrace_02�
F__inference_dense_63_layer_call_and_return_conditional_losses_11155624�
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
 zYtrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
&__inference_signature_wrapper_11155493input_1"�
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
Z	variables
[	keras_api
	\total
	]count"
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
+__inference_dense_60_layer_call_fn_11155554inputs"�
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
F__inference_dense_60_layer_call_and_return_conditional_losses_11155565inputs"�
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
+__inference_dense_61_layer_call_fn_11155574inputs"�
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
F__inference_dense_61_layer_call_and_return_conditional_losses_11155585inputs"�
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
+__inference_dense_62_layer_call_fn_11155594inputs"�
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
F__inference_dense_62_layer_call_and_return_conditional_losses_11155605inputs"�
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
+__inference_dense_63_layer_call_fn_11155614inputs"�
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
F__inference_dense_63_layer_call_and_return_conditional_losses_11155624inputs"�
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
\0
]1"
trackable_list_wrapper
-
Z	variables"
_generic_user_object
:  (2total
:  (2count
4:2	2$Adam/base_model_15/dense_60/kernel/m
.:,2"Adam/base_model_15/dense_60/bias/m
4:2 2$Adam/base_model_15/dense_61/kernel/m
.:, 2"Adam/base_model_15/dense_61/bias/m
4:2 2$Adam/base_model_15/dense_62/kernel/m
.:,2"Adam/base_model_15/dense_62/bias/m
4:22$Adam/base_model_15/dense_63/kernel/m
.:,2"Adam/base_model_15/dense_63/bias/m
4:2	2$Adam/base_model_15/dense_60/kernel/v
.:,2"Adam/base_model_15/dense_60/bias/v
4:2 2$Adam/base_model_15/dense_61/kernel/v
.:, 2"Adam/base_model_15/dense_61/bias/v
4:2 2$Adam/base_model_15/dense_62/kernel/v
.:,2"Adam/base_model_15/dense_62/bias/v
4:22$Adam/base_model_15/dense_63/kernel/v
.:,2"Adam/base_model_15/dense_63/bias/v�
#__inference__wrapped_model_11155285q0�-
&�#
!�
input_1���������	
� "3�0
.
output_1"�
output_1����������
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155464c0�-
&�#
!�
input_1���������	
� "%�"
�
0���������
� �
K__inference_base_model_15_layer_call_and_return_conditional_losses_11155545b/�,
%�"
 �
inputs���������	
� "%�"
�
0���������
� �
0__inference_base_model_15_layer_call_fn_11155379V0�-
&�#
!�
input_1���������	
� "�����������
0__inference_base_model_15_layer_call_fn_11155514U/�,
%�"
 �
inputs���������	
� "�����������
F__inference_dense_60_layer_call_and_return_conditional_losses_11155565\/�,
%�"
 �
inputs���������	
� "%�"
�
0���������
� ~
+__inference_dense_60_layer_call_fn_11155554O/�,
%�"
 �
inputs���������	
� "�����������
F__inference_dense_61_layer_call_and_return_conditional_losses_11155585\/�,
%�"
 �
inputs���������
� "%�"
�
0��������� 
� ~
+__inference_dense_61_layer_call_fn_11155574O/�,
%�"
 �
inputs���������
� "���������� �
F__inference_dense_62_layer_call_and_return_conditional_losses_11155605\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� ~
+__inference_dense_62_layer_call_fn_11155594O/�,
%�"
 �
inputs��������� 
� "�����������
F__inference_dense_63_layer_call_and_return_conditional_losses_11155624\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� ~
+__inference_dense_63_layer_call_fn_11155614O/�,
%�"
 �
inputs���������
� "�����������
&__inference_signature_wrapper_11155493|;�8
� 
1�.
,
input_1!�
input_1���������	"3�0
.
output_1"�
output_1���������