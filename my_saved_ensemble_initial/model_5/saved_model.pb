чд
бƒ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Ѕ
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
executor_typestring И®
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02unknown8це
Ъ
!Adam/base_model_5/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!Adam/base_model_5/dense_23/bias/v
У
5Adam/base_model_5/dense_23/bias/v/Read/ReadVariableOpReadVariableOp!Adam/base_model_5/dense_23/bias/v*
_output_shapes
:	*
dtype0
Ґ
#Adam/base_model_5/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*4
shared_name%#Adam/base_model_5/dense_23/kernel/v
Ы
7Adam/base_model_5/dense_23/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/base_model_5/dense_23/kernel/v*
_output_shapes

:	*
dtype0
Ъ
!Adam/base_model_5/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_5/dense_22/bias/v
У
5Adam/base_model_5/dense_22/bias/v/Read/ReadVariableOpReadVariableOp!Adam/base_model_5/dense_22/bias/v*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_5/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/base_model_5/dense_22/kernel/v
Ы
7Adam/base_model_5/dense_22/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/base_model_5/dense_22/kernel/v*
_output_shapes

: *
dtype0
Ъ
!Adam/base_model_5/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/base_model_5/dense_21/bias/v
У
5Adam/base_model_5/dense_21/bias/v/Read/ReadVariableOpReadVariableOp!Adam/base_model_5/dense_21/bias/v*
_output_shapes
: *
dtype0
Ґ
#Adam/base_model_5/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/base_model_5/dense_21/kernel/v
Ы
7Adam/base_model_5/dense_21/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/base_model_5/dense_21/kernel/v*
_output_shapes

: *
dtype0
Ъ
!Adam/base_model_5/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_5/dense_20/bias/v
У
5Adam/base_model_5/dense_20/bias/v/Read/ReadVariableOpReadVariableOp!Adam/base_model_5/dense_20/bias/v*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_5/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*4
shared_name%#Adam/base_model_5/dense_20/kernel/v
Ы
7Adam/base_model_5/dense_20/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/base_model_5/dense_20/kernel/v*
_output_shapes

:	*
dtype0
Ъ
!Adam/base_model_5/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!Adam/base_model_5/dense_23/bias/m
У
5Adam/base_model_5/dense_23/bias/m/Read/ReadVariableOpReadVariableOp!Adam/base_model_5/dense_23/bias/m*
_output_shapes
:	*
dtype0
Ґ
#Adam/base_model_5/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*4
shared_name%#Adam/base_model_5/dense_23/kernel/m
Ы
7Adam/base_model_5/dense_23/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/base_model_5/dense_23/kernel/m*
_output_shapes

:	*
dtype0
Ъ
!Adam/base_model_5/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_5/dense_22/bias/m
У
5Adam/base_model_5/dense_22/bias/m/Read/ReadVariableOpReadVariableOp!Adam/base_model_5/dense_22/bias/m*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_5/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/base_model_5/dense_22/kernel/m
Ы
7Adam/base_model_5/dense_22/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/base_model_5/dense_22/kernel/m*
_output_shapes

: *
dtype0
Ъ
!Adam/base_model_5/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/base_model_5/dense_21/bias/m
У
5Adam/base_model_5/dense_21/bias/m/Read/ReadVariableOpReadVariableOp!Adam/base_model_5/dense_21/bias/m*
_output_shapes
: *
dtype0
Ґ
#Adam/base_model_5/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/base_model_5/dense_21/kernel/m
Ы
7Adam/base_model_5/dense_21/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/base_model_5/dense_21/kernel/m*
_output_shapes

: *
dtype0
Ъ
!Adam/base_model_5/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_5/dense_20/bias/m
У
5Adam/base_model_5/dense_20/bias/m/Read/ReadVariableOpReadVariableOp!Adam/base_model_5/dense_20/bias/m*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_5/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*4
shared_name%#Adam/base_model_5/dense_20/kernel/m
Ы
7Adam/base_model_5/dense_20/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/base_model_5/dense_20/kernel/m*
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
М
base_model_5/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_namebase_model_5/dense_23/bias
Е
.base_model_5/dense_23/bias/Read/ReadVariableOpReadVariableOpbase_model_5/dense_23/bias*
_output_shapes
:	*
dtype0
Ф
base_model_5/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*-
shared_namebase_model_5/dense_23/kernel
Н
0base_model_5/dense_23/kernel/Read/ReadVariableOpReadVariableOpbase_model_5/dense_23/kernel*
_output_shapes

:	*
dtype0
М
base_model_5/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebase_model_5/dense_22/bias
Е
.base_model_5/dense_22/bias/Read/ReadVariableOpReadVariableOpbase_model_5/dense_22/bias*
_output_shapes
:*
dtype0
Ф
base_model_5/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_namebase_model_5/dense_22/kernel
Н
0base_model_5/dense_22/kernel/Read/ReadVariableOpReadVariableOpbase_model_5/dense_22/kernel*
_output_shapes

: *
dtype0
М
base_model_5/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebase_model_5/dense_21/bias
Е
.base_model_5/dense_21/bias/Read/ReadVariableOpReadVariableOpbase_model_5/dense_21/bias*
_output_shapes
: *
dtype0
Ф
base_model_5/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_namebase_model_5/dense_21/kernel
Н
0base_model_5/dense_21/kernel/Read/ReadVariableOpReadVariableOpbase_model_5/dense_21/kernel*
_output_shapes

: *
dtype0
М
base_model_5/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebase_model_5/dense_20/bias
Е
.base_model_5/dense_20/bias/Read/ReadVariableOpReadVariableOpbase_model_5/dense_20/bias*
_output_shapes
:*
dtype0
Ф
base_model_5/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*-
shared_namebase_model_5/dense_20/kernel
Н
0base_model_5/dense_20/kernel/Read/ReadVariableOpReadVariableOpbase_model_5/dense_20/kernel*
_output_shapes

:	*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:€€€€€€€€€	*
dtype0*
shape:€€€€€€€€€	
Ђ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1base_model_5/dense_20/kernelbase_model_5/dense_20/biasbase_model_5/dense_21/kernelbase_model_5/dense_21/biasbase_model_5/dense_22/kernelbase_model_5/dense_22/biasbase_model_5/dense_23/kernelbase_model_5/dense_23/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_9778129

NoOpNoOp
Ћ5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ж5
valueь4Bщ4 Bт4
В
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
∞
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
¶
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias*
¶
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias*
¶
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias*
¶
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias*
‘
7iter

8beta_1

9beta_2
	:decay
;learning_ratem^m_m`mambmcmdmevfvgvhvivjvkvlvm*

<serving_default* 
\V
VARIABLE_VALUEbase_model_5/dense_20/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbase_model_5/dense_20/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbase_model_5/dense_21/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbase_model_5/dense_21/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbase_model_5/dense_22/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbase_model_5/dense_22/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbase_model_5/dense_23/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbase_model_5/dense_23/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
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
У
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
У
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
У
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
У
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
y
VARIABLE_VALUE#Adam/base_model_5/dense_20/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_5/dense_20/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_5/dense_21/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_5/dense_21/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_5/dense_22/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_5/dense_22/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_5/dense_23/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_5/dense_23/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_5/dense_20/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_5/dense_20/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_5/dense_21/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_5/dense_21/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_5/dense_22/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_5/dense_22/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_5/dense_23/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_5/dense_23/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ґ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0base_model_5/dense_20/kernel/Read/ReadVariableOp.base_model_5/dense_20/bias/Read/ReadVariableOp0base_model_5/dense_21/kernel/Read/ReadVariableOp.base_model_5/dense_21/bias/Read/ReadVariableOp0base_model_5/dense_22/kernel/Read/ReadVariableOp.base_model_5/dense_22/bias/Read/ReadVariableOp0base_model_5/dense_23/kernel/Read/ReadVariableOp.base_model_5/dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7Adam/base_model_5/dense_20/kernel/m/Read/ReadVariableOp5Adam/base_model_5/dense_20/bias/m/Read/ReadVariableOp7Adam/base_model_5/dense_21/kernel/m/Read/ReadVariableOp5Adam/base_model_5/dense_21/bias/m/Read/ReadVariableOp7Adam/base_model_5/dense_22/kernel/m/Read/ReadVariableOp5Adam/base_model_5/dense_22/bias/m/Read/ReadVariableOp7Adam/base_model_5/dense_23/kernel/m/Read/ReadVariableOp5Adam/base_model_5/dense_23/bias/m/Read/ReadVariableOp7Adam/base_model_5/dense_20/kernel/v/Read/ReadVariableOp5Adam/base_model_5/dense_20/bias/v/Read/ReadVariableOp7Adam/base_model_5/dense_21/kernel/v/Read/ReadVariableOp5Adam/base_model_5/dense_21/bias/v/Read/ReadVariableOp7Adam/base_model_5/dense_22/kernel/v/Read/ReadVariableOp5Adam/base_model_5/dense_22/bias/v/Read/ReadVariableOp7Adam/base_model_5/dense_23/kernel/v/Read/ReadVariableOp5Adam/base_model_5/dense_23/bias/v/Read/ReadVariableOpConst*,
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
GPU 2J 8В *)
f$R"
 __inference__traced_save_9778376
±	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebase_model_5/dense_20/kernelbase_model_5/dense_20/biasbase_model_5/dense_21/kernelbase_model_5/dense_21/biasbase_model_5/dense_22/kernelbase_model_5/dense_22/biasbase_model_5/dense_23/kernelbase_model_5/dense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount#Adam/base_model_5/dense_20/kernel/m!Adam/base_model_5/dense_20/bias/m#Adam/base_model_5/dense_21/kernel/m!Adam/base_model_5/dense_21/bias/m#Adam/base_model_5/dense_22/kernel/m!Adam/base_model_5/dense_22/bias/m#Adam/base_model_5/dense_23/kernel/m!Adam/base_model_5/dense_23/bias/m#Adam/base_model_5/dense_20/kernel/v!Adam/base_model_5/dense_20/bias/v#Adam/base_model_5/dense_21/kernel/v!Adam/base_model_5/dense_21/bias/v#Adam/base_model_5/dense_22/kernel/v!Adam/base_model_5/dense_22/bias/v#Adam/base_model_5/dense_23/kernel/v!Adam/base_model_5/dense_23/bias/v*+
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
GPU 2J 8В *,
f'R%
#__inference__traced_restore_9778479ьЌ
ƒ
Ч
*__inference_dense_22_layer_call_fn_9778230

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_9777973o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ё#
ј
I__inference_base_model_5_layer_call_and_return_conditional_losses_9778181

inputs9
'dense_20_matmul_readvariableop_resource:	6
(dense_20_biasadd_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource: 6
(dense_21_biasadd_readvariableop_resource: 9
'dense_22_matmul_readvariableop_resource: 6
(dense_22_biasadd_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:	6
(dense_23_biasadd_readvariableop_resource:	
identityИҐdense_20/BiasAdd/ReadVariableOpҐdense_20/MatMul/ReadVariableOpҐdense_21/BiasAdd/ReadVariableOpҐdense_21/MatMul/ReadVariableOpҐdense_22/BiasAdd/ReadVariableOpҐdense_22/MatMul/ReadVariableOpҐdense_23/BiasAdd/ReadVariableOpҐdense_23/MatMul/ReadVariableOpЖ
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Р
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Р
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0Р
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	Д
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0С
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	h
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	“
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
Ь

ц
E__inference_dense_21_layer_call_and_return_conditional_losses_9778221

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ƒ
Ч
*__inference_dense_20_layer_call_fn_9778190

inputs
unknown:	
	unknown_0:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_9777939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
»	
ц
E__inference_dense_23_layer_call_and_return_conditional_losses_9777989

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь

ц
E__inference_dense_20_layer_call_and_return_conditional_losses_9777939

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
ЅE
©
 __inference__traced_save_9778376
file_prefix;
7savev2_base_model_5_dense_20_kernel_read_readvariableop9
5savev2_base_model_5_dense_20_bias_read_readvariableop;
7savev2_base_model_5_dense_21_kernel_read_readvariableop9
5savev2_base_model_5_dense_21_bias_read_readvariableop;
7savev2_base_model_5_dense_22_kernel_read_readvariableop9
5savev2_base_model_5_dense_22_bias_read_readvariableop;
7savev2_base_model_5_dense_23_kernel_read_readvariableop9
5savev2_base_model_5_dense_23_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_adam_base_model_5_dense_20_kernel_m_read_readvariableop@
<savev2_adam_base_model_5_dense_20_bias_m_read_readvariableopB
>savev2_adam_base_model_5_dense_21_kernel_m_read_readvariableop@
<savev2_adam_base_model_5_dense_21_bias_m_read_readvariableopB
>savev2_adam_base_model_5_dense_22_kernel_m_read_readvariableop@
<savev2_adam_base_model_5_dense_22_bias_m_read_readvariableopB
>savev2_adam_base_model_5_dense_23_kernel_m_read_readvariableop@
<savev2_adam_base_model_5_dense_23_bias_m_read_readvariableopB
>savev2_adam_base_model_5_dense_20_kernel_v_read_readvariableop@
<savev2_adam_base_model_5_dense_20_bias_v_read_readvariableopB
>savev2_adam_base_model_5_dense_21_kernel_v_read_readvariableop@
<savev2_adam_base_model_5_dense_21_bias_v_read_readvariableopB
>savev2_adam_base_model_5_dense_22_kernel_v_read_readvariableop@
<savev2_adam_base_model_5_dense_22_bias_v_read_readvariableopB
>savev2_adam_base_model_5_dense_23_kernel_v_read_readvariableop@
<savev2_adam_base_model_5_dense_23_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: џ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Д
valueъBч B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH≠
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B П
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_base_model_5_dense_20_kernel_read_readvariableop5savev2_base_model_5_dense_20_bias_read_readvariableop7savev2_base_model_5_dense_21_kernel_read_readvariableop5savev2_base_model_5_dense_21_bias_read_readvariableop7savev2_base_model_5_dense_22_kernel_read_readvariableop5savev2_base_model_5_dense_22_bias_read_readvariableop7savev2_base_model_5_dense_23_kernel_read_readvariableop5savev2_base_model_5_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_adam_base_model_5_dense_20_kernel_m_read_readvariableop<savev2_adam_base_model_5_dense_20_bias_m_read_readvariableop>savev2_adam_base_model_5_dense_21_kernel_m_read_readvariableop<savev2_adam_base_model_5_dense_21_bias_m_read_readvariableop>savev2_adam_base_model_5_dense_22_kernel_m_read_readvariableop<savev2_adam_base_model_5_dense_22_bias_m_read_readvariableop>savev2_adam_base_model_5_dense_23_kernel_m_read_readvariableop<savev2_adam_base_model_5_dense_23_bias_m_read_readvariableop>savev2_adam_base_model_5_dense_20_kernel_v_read_readvariableop<savev2_adam_base_model_5_dense_20_bias_v_read_readvariableop>savev2_adam_base_model_5_dense_21_kernel_v_read_readvariableop<savev2_adam_base_model_5_dense_21_bias_v_read_readvariableop>savev2_adam_base_model_5_dense_22_kernel_v_read_readvariableop<savev2_adam_base_model_5_dense_22_bias_v_read_readvariableop>savev2_adam_base_model_5_dense_23_kernel_v_read_readvariableop<savev2_adam_base_model_5_dense_23_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*з
_input_shapes’
“: :	:: : : ::	:	: : : : : : : :	:: : : ::	:	:	:: : : ::	:	: 2(
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

:	: 

_output_shapes
:	:	
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

:	: 

_output_shapes
:	:$ 

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

:	: 

_output_shapes
:	: 

_output_shapes
: 
»	
љ
.__inference_base_model_5_layer_call_fn_9778150

inputs
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:	
	unknown_6:	
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_base_model_5_layer_call_and_return_conditional_losses_9777996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
Б
Д
I__inference_base_model_5_layer_call_and_return_conditional_losses_9777996

inputs"
dense_20_9777940:	
dense_20_9777942:"
dense_21_9777957: 
dense_21_9777959: "
dense_22_9777974: 
dense_22_9777976:"
dense_23_9777990:	
dense_23_9777992:	
identityИҐ dense_20/StatefulPartitionedCallҐ dense_21/StatefulPartitionedCallҐ dense_22/StatefulPartitionedCallҐ dense_23/StatefulPartitionedCallу
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_9777940dense_20_9777942*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_9777939Ц
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_9777957dense_21_9777959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_9777956Ц
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_9777974dense_22_9777976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_9777973Ц
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_9777990dense_23_9777992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_9777989x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	“
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
Ћ	
Њ
.__inference_base_model_5_layer_call_fn_9778015
input_1
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:	
	unknown_6:	
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_base_model_5_layer_call_and_return_conditional_losses_9777996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€	
!
_user_specified_name	input_1
Ь

ц
E__inference_dense_20_layer_call_and_return_conditional_losses_9778201

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
ƒ
Ч
*__inference_dense_23_layer_call_fn_9778250

inputs
unknown:	
	unknown_0:	
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_9777989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь

ц
E__inference_dense_22_layer_call_and_return_conditional_losses_9778241

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ƒ
Ч
*__inference_dense_21_layer_call_fn_9778210

inputs
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_9777956o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь

ц
E__inference_dense_21_layer_call_and_return_conditional_losses_9777956

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»	
ц
E__inference_dense_23_layer_call_and_return_conditional_losses_9778260

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Д
Е
I__inference_base_model_5_layer_call_and_return_conditional_losses_9778100
input_1"
dense_20_9778079:	
dense_20_9778081:"
dense_21_9778084: 
dense_21_9778086: "
dense_22_9778089: 
dense_22_9778091:"
dense_23_9778094:	
dense_23_9778096:	
identityИҐ dense_20/StatefulPartitionedCallҐ dense_21/StatefulPartitionedCallҐ dense_22/StatefulPartitionedCallҐ dense_23/StatefulPartitionedCallф
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_20_9778079dense_20_9778081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_9777939Ц
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_9778084dense_21_9778086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_9777956Ц
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_9778089dense_22_9778091*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_9777973Ц
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_9778094dense_23_9778096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_9777989x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	“
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€	
!
_user_specified_name	input_1
Ь

ц
E__inference_dense_22_layer_call_and_return_conditional_losses_9777973

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
К
Й
#__inference__traced_restore_9778479
file_prefix?
-assignvariableop_base_model_5_dense_20_kernel:	;
-assignvariableop_1_base_model_5_dense_20_bias:A
/assignvariableop_2_base_model_5_dense_21_kernel: ;
-assignvariableop_3_base_model_5_dense_21_bias: A
/assignvariableop_4_base_model_5_dense_22_kernel: ;
-assignvariableop_5_base_model_5_dense_22_bias:A
/assignvariableop_6_base_model_5_dense_23_kernel:	;
-assignvariableop_7_base_model_5_dense_23_bias:	&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: I
7assignvariableop_15_adam_base_model_5_dense_20_kernel_m:	C
5assignvariableop_16_adam_base_model_5_dense_20_bias_m:I
7assignvariableop_17_adam_base_model_5_dense_21_kernel_m: C
5assignvariableop_18_adam_base_model_5_dense_21_bias_m: I
7assignvariableop_19_adam_base_model_5_dense_22_kernel_m: C
5assignvariableop_20_adam_base_model_5_dense_22_bias_m:I
7assignvariableop_21_adam_base_model_5_dense_23_kernel_m:	C
5assignvariableop_22_adam_base_model_5_dense_23_bias_m:	I
7assignvariableop_23_adam_base_model_5_dense_20_kernel_v:	C
5assignvariableop_24_adam_base_model_5_dense_20_bias_v:I
7assignvariableop_25_adam_base_model_5_dense_21_kernel_v: C
5assignvariableop_26_adam_base_model_5_dense_21_bias_v: I
7assignvariableop_27_adam_base_model_5_dense_22_kernel_v: C
5assignvariableop_28_adam_base_model_5_dense_22_bias_v:I
7assignvariableop_29_adam_base_model_5_dense_23_kernel_v:	C
5assignvariableop_30_adam_base_model_5_dense_23_bias_v:	
identity_32ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Д
valueъBч B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH∞
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ѕ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOpAssignVariableOp-assignvariableop_base_model_5_dense_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_1AssignVariableOp-assignvariableop_1_base_model_5_dense_20_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp/assignvariableop_2_base_model_5_dense_21_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_3AssignVariableOp-assignvariableop_3_base_model_5_dense_21_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_4AssignVariableOp/assignvariableop_4_base_model_5_dense_22_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_5AssignVariableOp-assignvariableop_5_base_model_5_dense_22_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_6AssignVariableOp/assignvariableop_6_base_model_5_dense_23_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_7AssignVariableOp-assignvariableop_7_base_model_5_dense_23_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_15AssignVariableOp7assignvariableop_15_adam_base_model_5_dense_20_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adam_base_model_5_dense_20_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adam_base_model_5_dense_21_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adam_base_model_5_dense_21_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_base_model_5_dense_22_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_base_model_5_dense_22_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_base_model_5_dense_23_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_base_model_5_dense_23_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adam_base_model_5_dense_20_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_base_model_5_dense_20_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_base_model_5_dense_21_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_base_model_5_dense_21_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_base_model_5_dense_22_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_base_model_5_dense_22_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_base_model_5_dense_23_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_base_model_5_dense_23_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 щ
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ж
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
Щ,
к
"__inference__wrapped_model_9777921
input_1F
4base_model_5_dense_20_matmul_readvariableop_resource:	C
5base_model_5_dense_20_biasadd_readvariableop_resource:F
4base_model_5_dense_21_matmul_readvariableop_resource: C
5base_model_5_dense_21_biasadd_readvariableop_resource: F
4base_model_5_dense_22_matmul_readvariableop_resource: C
5base_model_5_dense_22_biasadd_readvariableop_resource:F
4base_model_5_dense_23_matmul_readvariableop_resource:	C
5base_model_5_dense_23_biasadd_readvariableop_resource:	
identityИҐ,base_model_5/dense_20/BiasAdd/ReadVariableOpҐ+base_model_5/dense_20/MatMul/ReadVariableOpҐ,base_model_5/dense_21/BiasAdd/ReadVariableOpҐ+base_model_5/dense_21/MatMul/ReadVariableOpҐ,base_model_5/dense_22/BiasAdd/ReadVariableOpҐ+base_model_5/dense_22/MatMul/ReadVariableOpҐ,base_model_5/dense_23/BiasAdd/ReadVariableOpҐ+base_model_5/dense_23/MatMul/ReadVariableOp†
+base_model_5/dense_20/MatMul/ReadVariableOpReadVariableOp4base_model_5_dense_20_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0Ц
base_model_5/dense_20/MatMulMatMulinput_13base_model_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
,base_model_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp5base_model_5_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Є
base_model_5/dense_20/BiasAddBiasAdd&base_model_5/dense_20/MatMul:product:04base_model_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
base_model_5/dense_20/ReluRelu&base_model_5/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€†
+base_model_5/dense_21/MatMul/ReadVariableOpReadVariableOp4base_model_5_dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
base_model_5/dense_21/MatMulMatMul(base_model_5/dense_20/Relu:activations:03base_model_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
,base_model_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp5base_model_5_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Є
base_model_5/dense_21/BiasAddBiasAdd&base_model_5/dense_21/MatMul:product:04base_model_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ |
base_model_5/dense_21/ReluRelu&base_model_5/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ †
+base_model_5/dense_22/MatMul/ReadVariableOpReadVariableOp4base_model_5_dense_22_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
base_model_5/dense_22/MatMulMatMul(base_model_5/dense_21/Relu:activations:03base_model_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
,base_model_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp5base_model_5_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Є
base_model_5/dense_22/BiasAddBiasAdd&base_model_5/dense_22/MatMul:product:04base_model_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
base_model_5/dense_22/ReluRelu&base_model_5/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€†
+base_model_5/dense_23/MatMul/ReadVariableOpReadVariableOp4base_model_5_dense_23_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0Ј
base_model_5/dense_23/MatMulMatMul(base_model_5/dense_22/Relu:activations:03base_model_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	Ю
,base_model_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp5base_model_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Є
base_model_5/dense_23/BiasAddBiasAdd&base_model_5/dense_23/MatMul:product:04base_model_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	u
IdentityIdentity&base_model_5/dense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	Ї
NoOpNoOp-^base_model_5/dense_20/BiasAdd/ReadVariableOp,^base_model_5/dense_20/MatMul/ReadVariableOp-^base_model_5/dense_21/BiasAdd/ReadVariableOp,^base_model_5/dense_21/MatMul/ReadVariableOp-^base_model_5/dense_22/BiasAdd/ReadVariableOp,^base_model_5/dense_22/MatMul/ReadVariableOp-^base_model_5/dense_23/BiasAdd/ReadVariableOp,^base_model_5/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 2\
,base_model_5/dense_20/BiasAdd/ReadVariableOp,base_model_5/dense_20/BiasAdd/ReadVariableOp2Z
+base_model_5/dense_20/MatMul/ReadVariableOp+base_model_5/dense_20/MatMul/ReadVariableOp2\
,base_model_5/dense_21/BiasAdd/ReadVariableOp,base_model_5/dense_21/BiasAdd/ReadVariableOp2Z
+base_model_5/dense_21/MatMul/ReadVariableOp+base_model_5/dense_21/MatMul/ReadVariableOp2\
,base_model_5/dense_22/BiasAdd/ReadVariableOp,base_model_5/dense_22/BiasAdd/ReadVariableOp2Z
+base_model_5/dense_22/MatMul/ReadVariableOp+base_model_5/dense_22/MatMul/ReadVariableOp2\
,base_model_5/dense_23/BiasAdd/ReadVariableOp,base_model_5/dense_23/BiasAdd/ReadVariableOp2Z
+base_model_5/dense_23/MatMul/ReadVariableOp+base_model_5/dense_23/MatMul/ReadVariableOp:P L
'
_output_shapes
:€€€€€€€€€	
!
_user_specified_name	input_1
Ы	
µ
%__inference_signature_wrapper_9778129
input_1
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:	
	unknown_6:	
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_9777921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€	
!
_user_specified_name	input_1"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_defaultЧ
;
input_10
serving_default_input_1:0€€€€€€€€€	<
output_10
StatefulPartitionedCall:0€€€€€€€€€	tensorflow/serving/predict:єn
Ч
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
 
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
Љ
trace_0
trace_12Е
.__inference_base_model_5_layer_call_fn_9778015
.__inference_base_model_5_layer_call_fn_9778150Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ztrace_0ztrace_1
т
trace_0
trace_12ї
I__inference_base_model_5_layer_call_and_return_conditional_losses_9778181
I__inference_base_model_5_layer_call_and_return_conditional_losses_9778100Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ztrace_0ztrace_1
ЌB 
"__inference__wrapped_model_9777921input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ї
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
г
7iter

8beta_1

9beta_2
	:decay
;learning_ratem^m_m`mambmcmdmevfvgvhvivjvkvlvm"
	optimizer
,
<serving_default"
signature_map
.:,	2base_model_5/dense_20/kernel
(:&2base_model_5/dense_20/bias
.:, 2base_model_5/dense_21/kernel
(:& 2base_model_5/dense_21/bias
.:, 2base_model_5/dense_22/kernel
(:&2base_model_5/dense_22/bias
.:,	2base_model_5/dense_23/kernel
(:&	2base_model_5/dense_23/bias
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
гBа
.__inference_base_model_5_layer_call_fn_9778015input_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
вBя
.__inference_base_model_5_layer_call_fn_9778150inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
I__inference_base_model_5_layer_call_and_return_conditional_losses_9778181inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
I__inference_base_model_5_layer_call_and_return_conditional_losses_9778100input_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
Ctrace_02—
*__inference_dense_20_layer_call_fn_9778190Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zCtrace_0
Й
Dtrace_02м
E__inference_dense_20_layer_call_and_return_conditional_losses_9778201Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
Jtrace_02—
*__inference_dense_21_layer_call_fn_9778210Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zJtrace_0
Й
Ktrace_02м
E__inference_dense_21_layer_call_and_return_conditional_losses_9778221Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
Qtrace_02—
*__inference_dense_22_layer_call_fn_9778230Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zQtrace_0
Й
Rtrace_02м
E__inference_dense_22_layer_call_and_return_conditional_losses_9778241Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
Xtrace_02—
*__inference_dense_23_layer_call_fn_9778250Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zXtrace_0
Й
Ytrace_02м
E__inference_dense_23_layer_call_and_return_conditional_losses_9778260Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zYtrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ћB…
%__inference_signature_wrapper_9778129input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_dense_20_layer_call_fn_9778190inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_dense_20_layer_call_and_return_conditional_losses_9778201inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_dense_21_layer_call_fn_9778210inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_dense_21_layer_call_and_return_conditional_losses_9778221inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_dense_22_layer_call_fn_9778230inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_dense_22_layer_call_and_return_conditional_losses_9778241inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_dense_23_layer_call_fn_9778250inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_dense_23_layer_call_and_return_conditional_losses_9778260inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
3:1	2#Adam/base_model_5/dense_20/kernel/m
-:+2!Adam/base_model_5/dense_20/bias/m
3:1 2#Adam/base_model_5/dense_21/kernel/m
-:+ 2!Adam/base_model_5/dense_21/bias/m
3:1 2#Adam/base_model_5/dense_22/kernel/m
-:+2!Adam/base_model_5/dense_22/bias/m
3:1	2#Adam/base_model_5/dense_23/kernel/m
-:+	2!Adam/base_model_5/dense_23/bias/m
3:1	2#Adam/base_model_5/dense_20/kernel/v
-:+2!Adam/base_model_5/dense_20/bias/v
3:1 2#Adam/base_model_5/dense_21/kernel/v
-:+ 2!Adam/base_model_5/dense_21/bias/v
3:1 2#Adam/base_model_5/dense_22/kernel/v
-:+2!Adam/base_model_5/dense_22/bias/v
3:1	2#Adam/base_model_5/dense_23/kernel/v
-:+	2!Adam/base_model_5/dense_23/bias/vЧ
"__inference__wrapped_model_9777921q0Ґ-
&Ґ#
!К
input_1€€€€€€€€€	
™ "3™0
.
output_1"К
output_1€€€€€€€€€	∞
I__inference_base_model_5_layer_call_and_return_conditional_losses_9778100c0Ґ-
&Ґ#
!К
input_1€€€€€€€€€	
™ "%Ґ"
К
0€€€€€€€€€	
Ъ ѓ
I__inference_base_model_5_layer_call_and_return_conditional_losses_9778181b/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%Ґ"
К
0€€€€€€€€€	
Ъ И
.__inference_base_model_5_layer_call_fn_9778015V0Ґ-
&Ґ#
!К
input_1€€€€€€€€€	
™ "К€€€€€€€€€	З
.__inference_base_model_5_layer_call_fn_9778150U/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "К€€€€€€€€€	•
E__inference_dense_20_layer_call_and_return_conditional_losses_9778201\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_dense_20_layer_call_fn_9778190O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "К€€€€€€€€€•
E__inference_dense_21_layer_call_and_return_conditional_losses_9778221\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ }
*__inference_dense_21_layer_call_fn_9778210O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€ •
E__inference_dense_22_layer_call_and_return_conditional_losses_9778241\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_dense_22_layer_call_fn_9778230O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€•
E__inference_dense_23_layer_call_and_return_conditional_losses_9778260\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€	
Ъ }
*__inference_dense_23_layer_call_fn_9778250O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€	•
%__inference_signature_wrapper_9778129|;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€	"3™0
.
output_1"К
output_1€€€€€€€€€	