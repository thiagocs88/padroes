getwd()

#Leitura dos Dados
dados <- read.table("telescope.txt",header=TRUE,sep=",")

summary(dados)  #Teste, se importou corretamente a tabela

head(dados) #Primeiras linhas

str(dados) #Resumo das características

#PCA: Análise dos Componentes Principais
principais <- princomp(dados[,-11], cor = FALSE, scores = TRUE)

principais

#resumo
summary(principais)

plot(principais) #É chocante, mas as 5 primeiras variáveis registram mais de 
                 # + de 96% de toda a variabilidade dos dados

#Novo conjunto de dados oriundo do PCA
dadospca <- dados[,c(1,2,3,4,5,11)]

head(dadospca) #Coluna de rótulo agora é a 6

#Data Pre-Processing and Visualization Functions for Classification
library(dprep) #Contém função Relief

#Seleção de Características com o Relief
relief(dados,500,0.01,repet=3) #limiar=0,01, amostra de 1000 elementos, repetido 3 vezes

#Novo conjunto das variáveis mais "pesadas" (relief)
dadosrelief <- dados[,c(9,4,5,3,2,11)]

head(dadosrelief) #Teste

#Pacotes para teste com os Classificadores
library(caret) #C_lassification _A_nd _RE_gression _T_raining
library(e1071)
library(nnet)

#Início dos Testes com Classificadores

#Etapa 100: Classificadores com DADOS BRUTOS

TrainingDataIndexBrutos <- createDataPartition(dados$class, p=0.80, list = FALSE)
# Create Training Data 
trainingDataBrutos <- dados[TrainingDataIndexBrutos,]
testDataBrutos <- dados[-TrainingDataIndexBrutos,]
TrainingParameters <- trainControl(method = "cv", number = 5)
#Criando 5-Folds-Cross-Validation

#Teste SVM Linear
SVM100 <- train(class ~ ., data = trainingDataBrutos,
               method = "svmLinear2",
               trControl= TrainingParameters,
               #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
               preProcess = c("scale","center"),
               na.action = na.omit
)
SVM100
SVM100Predictions <-predict(SVM100, testDataBrutos)
# Create confusion matrix
cmSVM100 <-confusionMatrix(SVM100Predictions, testDataBrutos$class)
print(cmSVM100)


#Teste SVM Polinomial
SVM101 <- train(class ~ ., data = trainingDataBrutos,
                method = "svmPoly",
                trControl= TrainingParameters,
                #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
                preProcess = c("scale","center"),
                na.action = na.omit
)
SVM101
SVM101Predictions <-predict(SVM101, testDataBrutos)
# Create confusion matrix
cmSVM101 <-confusionMatrix(SVM101Predictions, testDataBrutos$class)
print(cmSVM101)



#Teste SVM Radial
SVM102 <- train(class ~ ., data = trainingDataBrutos,
                method = "svmRadial",
                trControl= TrainingParameters,
                #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
                preProcess = c("scale","center"),
                na.action = na.omit
)
SVM102
SVM102Predictions <-predict(SVM102, testDataBrutos)
# Create confusion matrix
cmSVM102 <-confusionMatrix(SVM102Predictions, testDataBrutos$class)
print(cmSVM102)


#Teste com Naive-Bayes
NaiveModel100 <- train(trainingDataBrutos[,-11], trainingDataBrutos$class, 
                     method = "nb",
                     preProcess=c("scale","center"),
                     trControl= TrainingParameters,
                     na.action = na.omit
)
NaiveModel100
#Predictions
Naive100Predictions <-predict(NaiveModel100, testDataBrutos, na.action = na.pass)
cmNaive100 <-confusionMatrix(Naive100Predictions, testDataBrutos$class)
cmNaive100


#Teste com Neural Network
NNModel100 <- train(trainingDataBrutos[,-11], trainingDataBrutos$class,
                 method = "nnet",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 na.action = na.omit
)
NNModel100
NN100Predictions <-predict(NNModel100, testDataBrutos)
# Create confusion matrix
cmNN100 <-confusionMatrix(NN100Predictions, testDataBrutos$class)
print(cmNN100)


#Etapa 200: Classificadores com DADOS PCA

TrainingDataIndexPCA <- createDataPartition(dadospca$class, p=0.80, list = FALSE)
# Create Training Data 
trainingDataPCA <- dadospca[TrainingDataIndexPCA,]
testDataPCA <- dadospca[-TrainingDataIndexPCA,]
#TrainingParameters <- trainControl(method = "cv", number = 5)
#Criando 5-Folds-Cross-Validation

#Teste SVM Linear
SVM200 <- train(class ~ ., data = trainingDataPCA,
                method = "svmLinear2",
                trControl= TrainingParameters,
                #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
                preProcess = c("scale","center"),
                na.action = na.omit
)
SVM200
SVM200Predictions <-predict(SVM200, testDataPCA)
# Create confusion matrix
cmSVM200 <-confusionMatrix(SVM200Predictions, testDataPCA$class)
print(cmSVM200)


#Teste SVM Polinomial
SVM201 <- train(class ~ ., data = trainingDataPCA,
                method = "svmPoly",
                trControl= TrainingParameters,
                #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
                preProcess = c("scale","center"),
                na.action = na.omit
)
SVM201
SVM201Predictions <-predict(SVM201, testDataPCA)
# Create confusion matrix
cmSVM201 <-confusionMatrix(SVM201Predictions, testDataPCA$class)
print(cmSVM201)



#Teste SVM Radial
SVM202 <- train(class ~ ., data = trainingDataPCA,
                method = "svmRadial",
                trControl= TrainingParameters,
                #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
                preProcess = c("scale","center"),
                na.action = na.omit
)
SVM202
SVM202Predictions <-predict(SVM202, testDataPCA)
# Create confusion matrix
cmSVM202 <-confusionMatrix(SVM202Predictions, testDataPCA$class)
print(cmSVM202)


#Teste com Naive-Bayes
NaiveModel200 <- train(trainingDataPCA[,-6], trainingDataPCA$class, 
                       method = "nb",
                       preProcess=c("scale","center"),
                       trControl= TrainingParameters,
                       na.action = na.omit
)
NaiveModel200
#Predictions
Naive200Predictions <-predict(NaiveModel200, testDataPCA, na.action = na.pass)
cmNaive200 <-confusionMatrix(Naive200Predictions, testDataPCA$class)
cmNaive200


#Teste com Neural Network
NNModel200 <- train(trainingDataPCA[,-6], trainingDataPCA$class,
                    method = "nnet",
                    trControl= TrainingParameters,
                    preProcess=c("scale","center"),
                    na.action = na.omit
)
NNModel200
NN200Predictions <-predict(NNModel200, testDataPCA)
# Create confusion matrix
cmNN200 <-confusionMatrix(NN200Predictions, testDataPCA$class)
print(cmNN200)



#Etapa 300: Classificadores com RELIEF

TrainingDataIndexRelief <- createDataPartition(dadosrelief$class, p=0.80, list = FALSE)
# Create Training Data 
trainingDataRelief <- dadosrelief[TrainingDataIndexRelief,]
testDataRelief <- dadosrelief[-TrainingDataIndexRelief,]
#TrainingParameters <- trainControl(method = "cv", number = 5)
#Criando 5-Folds-Cross-Validation

#Teste SVM Linear
SVM300 <- train(class ~ ., data = trainingDataRelief,
                method = "svmLinear2",
                trControl= TrainingParameters,
                #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
                preProcess = c("scale","center"),
                na.action = na.omit
)
SVM300
SVM300Predictions <-predict(SVM300, testDataRelief)
# Create confusion matrix
cmSVM300 <-confusionMatrix(SVM300Predictions, testDataRelief$class)
print(cmSVM300)


#Teste SVM Polinomial
SVM301 <- train(class ~ ., data = trainingDataRelief,
                method = "svmPoly",
                trControl= TrainingParameters,
                #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
                preProcess = c("scale","center"),
                na.action = na.omit
)
SVM301
SVM301Predictions <-predict(SVM301, testDataRelief)
# Create confusion matrix
cmSVM301 <-confusionMatrix(SVM301Predictions, testDataRelief$class)
print(cmSVM301)



#Teste SVM Radial
SVM302 <- train(class ~ ., data = trainingDataRelief,
                method = "svmRadial",
                trControl= TrainingParameters,
                #tuneGrid = a função escolhe os melhores parametros, pela cross-validation
                preProcess = c("scale","center"),
                na.action = na.omit
)
SVM302
SVM302Predictions <-predict(SVM302, testDataRelief)
# Create confusion matrix
cmSVM302 <-confusionMatrix(SVM302Predictions, testDataRelief$class)
print(cmSVM302)


#Teste com Naive-Bayes
NaiveModel300 <- train(trainingDataRelief[,-6], trainingDataRelief$class, 
                       method = "nb",
                       preProcess=c("scale","center"),
                       trControl= TrainingParameters,
                       na.action = na.omit
)
NaiveModel300
#Predictions
Naive300Predictions <-predict(NaiveModel300, testDataRelief, na.action = na.pass)
cmNaive300 <-confusionMatrix(Naive300Predictions, testDataRelief$class)
cmNaive300


#Teste com Neural Network
NNModel300 <- train(trainingDataRelief[,-6], trainingDataRelief$class,
                    method = "nnet",
                    trControl= TrainingParameters,
                    preProcess=c("scale","center"),
                    na.action = na.omit
)
NNModel300
NN300Predictions <-predict(NNModel300, testDataRelief)
# Create confusion matrix
cmNN300 <-confusionMatrix(NN300Predictions, testDataRelief$class)
print(cmNN300)
