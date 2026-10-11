# A matemática do dsgelab

Nota técnica do que o pacote resolve e de como o resolve. As fórmulas abaixo são as do código em `src/dsgelab`. Onde uma convenção usual da literatura produz outro número, o texto diz qual das duas está implementada e em que modelo as duas deixam de coincidir.

## 1. O objeto

Um modelo é uma função de resíduos

\[
f(x_{t+1}, x_t, \varepsilon_{t+1}; \theta) = 0,
\]

com tantas equações quanto variáveis. O vetor de estado divide-se em predeterminados \(k_t\), de dimensão \(n_k\), e não predeterminados \(u_t\), de dimensão \(n_u\),

\[
x_t = \begin{pmatrix} k_t \\ u_t \end{pmatrix}.
\]

A inovação \(\varepsilon_{t+1}\) é desconhecida em \(t\), com \(E_t \varepsilon_{t+1} = 0\). Um processo autorregressivo escreve-se

\[
\log a_{t+1} = \rho \log a_t + \varepsilon_{a,t+1}.
\]

Datar a inovação em \(t\), como se já estivesse no conjunto de informação quando \(u_t\) é escolhido, não altera a solução de primeira ordem, porque a equivalência à certeza iguala as duas datações nesse grau. Em segunda ordem a correção de risco sai identicamente zero, pois não resta incerteza dentro do período. O teste que fixa a convenção é o AR(1) lognormal da seção 7: a correção sobre a tecnologia tem de ser \(\sigma^2\). Sob a datação invertida ela é zero, e o resto da solução continua com aparência de modelo resolvido.

A mesma função \(f\) é linearizada, perturbada em ordem superior e empilhada na previsão perfeita. Uma trajetória determinística escrita à mão sobre outro conjunto de equações não entra no pacote, porque as duas deixariam de ser o mesmo modelo no momento em que uma delas fosse editada.

## 2. Estado estacionário

O estado estacionário determinístico \(\bar x\) resolve

\[
f(\bar x, \bar x, 0; \theta) = 0.
\]

A aceitação é pelo resíduo máximo, com tolerância padrão \(10^{-11}\). O critério de sucesso de `hybr` não entra na decisão. Esse método declara falha quando a jacobiana de diferenças finitas deixa de melhorar, e isso inclui o ponto de partida que já é a solução, com resíduo da ordem de \(10^{-14}\).

Variáveis marcadas como positivas são resolvidas em logaritmo. Se \(z_i = \log x_i\), o passo de Newton vive em \(z\) e \(x_i = e^{z_i}\) não atravessa zero. Um truncamento em zero achataria o resíduo e pararia o solver na fronteira.

Algumas equações dependem de uma grandeza que é função do próprio estado estacionário, como um custo de ajustamento centrado na riqueza estacionária. Substituir essa grandeza pela solução anula o termo e deixa a margem correspondente indeterminada. Ela viaja num contexto \(c\), atualizado por um laço externo

\[
c^{(n+1)} = \mathcal{C}(\bar x^{(n)}, c^{(n)}),
\]

que para quando \(c\) deixa de se mover, com tolerância padrão \(10^{-12}\). O contexto convergido segue com a solução e é herdado pela perturbação e pela transição.

## 3. Coordenadas da aproximação

A linearização e as ordens superiores trabalham no desvio reescalado

\[
x = \bar x + s \odot \hat y, \qquad
s_i = \begin{cases} |\bar x_i| & \text{se } \bar x_i \neq 0, \\ 1 & \text{se } \bar x_i = 0. \end{cases}
\]

A mudança de variável é afim. A hessiana em desvios é a hessiana em níveis multiplicada por \(s_i s_j\), e uma derivada exata fornecida em níveis converte-se com essa multiplicação. Uma log-linearização genuína, \(x_i = \bar x_i e^{\hat y_i}\), é não linear e não teria essa propriedade. O ramo \(s_i = 1\) é escolhido pelo nível estacionário, não pelo nome da variável. Um deslocador de média zero, com \(\bar x_i = 0\), seria anulado por uma regra que o escrevesse como \(\bar x_i(1+\hat y_i)\).

As linhas do lápis \((B, A)\) são em seguida divididas pelo máximo absoluto da linha, com piso \(10^{-12}\). A operação é uma semelhança do lápis. Os autovalores generalizados não mudam. O condicionamento muda, e é ele que decide se a contagem de Blanchard–Kahn é computável quando um estoque de capital de ordem dezenas convive com um deslocador de média zero.

## 4. Sistema de primeira ordem

No desvio reescalado, com \(f(\hat y_{t+1}, \hat y_t, \varepsilon_{t+1}) = 0\),

\[
A = \frac{\partial f}{\partial \hat y_{t+1}}, \quad
B = -\frac{\partial f}{\partial \hat y_t}, \quad
C = -\frac{\partial f}{\partial \varepsilon_{t+1}},
\]

e a forma canônica de Klein (2000) é

\[
A\, E_t \hat y_{t+1} = B \hat y_t + C \varepsilon_{t+1}.
\]

\(A\) pode ser singular. Equações estáticas põem linhas nulas nela, e nada no solver a inverte.

Se o modelo foi declarado em texto, a jacobiana é a derivada simbólica, convertida pela escala \(s\). Caso contrário, cada coluna é uma diferença central com passo \(h = 10^{-6}\),

\[
\frac{\partial f}{\partial \hat y_j}
  \approx \frac{f(h e_j) - f(-h e_j)}{2h}.
\]

A diferença para a frente é de primeira ordem, e o erro de truncamento entra nos autovalores dos quais a contagem de determinação é lida.

## 5. Solução de Klein

O lápis \((B, A)\) é levado à forma de Schur generalizada complexa, ordenada de modo que os autovalores de módulo menor que um ocupem o bloco inicial (Moler e Stewart, 1973; Klein, 2000). A forma real produz blocos \(2\times 2\) quando dois autovalores generalizados se aproximam em módulo, e esses blocos misturam os subespaços estável e instável. Vários AR(1) com persistências próximas são exatamente esse caso.

Com `ordqz(B, A)`, \(S\) é a forma de Schur de \(B\) e \(T\) a de \(A\). Os autovalores generalizados são \(\lambda = |\alpha|/|\beta|\), com \(|\beta|\) limitado inferiormente por \(10^{-9}\) para não dividir por zero num autovalor infinito. Um autovalor conta como estável quando \(\lambda < 1 - 10^{-9}\).

A condição de Blanchard e Kahn (1980) é imposta. O número de autovalores estáveis tem de ser igual a \(n_k\). Menos raízes estáveis significa que não há equilíbrio que não exploda. Mais raízes estáveis significa um contínuo de soluções. Nos dois casos \(F\) e \(P\) seriam contas sem interpretação, e a função levanta erro em vez de devolvê-las. A margem entre a maior raiz estável e a menor instável fica disponível à parte, porque uma raiz em \(0{,}9999\) passa no mesmo teste que uma raiz em \(0{,}8\) e troca de classificação sob uma perturbação de parâmetro pequena demais para aparecer numa varredura grosseira.

Particionando \(Z\), a matriz dos vetores de Schur, em blocos de \(n_k\) e \(n_u\),

\[
F = Z_{21} Z_{11}^{-1}, \qquad
P = Z_{11}\, T_{11}^{-1} S_{11}\, Z_{11}^{-1},
\]

tomadas a parte real. \(Z_{11}\) singular significa que o subespaço estável não gera os estados, e não existe mapa de estado para controle. A solução de primeira ordem é

\[
u_t = F k_t, \qquad k_{t+1} = P k_t + Q \varepsilon_{t+1}.
\]

### O carregamento dos choques

\(Q\) não é lido da transformação QZ. A fórmula de livro

\[
Q = Z_{11} T_{11}^{-1} (Q_z^{\mathrm H} C)_1
\]

resolve \(A x_{t+1} = B x_t + C\varepsilon\) como se toda equação valesse ex post. Com a inovação datada em \(t+1\), uma equação que contém um salto em \(t+1\) vale só em esperança. O salto responde a \(\varepsilon_{t+1}\), e a esperança dessa resposta em \(t\) é zero. Impor a linha ex post acrescenta uma restrição falsa. O sistema para \(Q\) fica inconsistente, e a QZ devolve um compromisso entre o carregamento verdadeiro e a restrição falsa.

Dois AR(1) independentes e uma curva de Phillips bastam para ver o efeito. O carregamento dos dois processos tem de ser a identidade. A fórmula de livro devolve uma matriz densa, com termos fora da diagonal da ordem de \(0{,}12\). Os autovalores e \(P\) continuam certos. Momentos, decomposição de variância e correção de risco, que passam todos por \(Q\), ficam errados. O erro some no RBC de utilidade logarítmica e depreciação total, porque a Euler desse caso vale ex post. Uma suíte construída só sobre esse modelo não o detecta.

O código usa as linhas que valem ex post, as que não têm coeficiente sobre um salto em \(t+1\). São a transição de cada processo exógeno, por onde a inovação entra, e a identidade de acumulação de cada estado endógeno, que força a linha correspondente de \(Q\) a zero. Um estoque escolhido em \(t\) não responde a um choque de \(t+1\). Sobre esse conjunto de linhas,

\[
M Q = C_{\text{linhas}}, \qquad
M = A_{\text{linhas},\,k} + A_{\text{linhas},\,u} F.
\]

O segundo termo é zero por construção nessas linhas. \(M\) tem posto \(n_k\) exatamente quando cada predeterminado tem lei de movimento. Posto menor significa um estado cuja lei falta ou foi escrita com um salto em \(t+1\). O resíduo \(\|MQ - C_{\text{linhas}}\|_\infty\) é reportado e tem de estar no arredondamento. Um resíduo material, numa equação sem salto em \(t+1\), aparece na prática quando o único termo futuro é um estado exógeno, como \(q_t = \beta f(a_{t+1})\). O termo futuro precisa ser declarado como salto, com a sua equação de definição, para a linha expectacional ser marcada como expectacional.

## 6. Impulsos, momentos e decomposição

Um impulso a uma inovação nomeada aplica \(k_0 = Q e\, m\), com \(m\) o tamanho, e propaga \(k_{t+1} = P k_t\). Um impulso a um estado desloca esse estado em \(t = 0\), que é o contrafactual sobre um nível exógeno e não sobre a sua inovação. Um vetor de impulso arbitrário cobre o caso ortogonalizado, a coluna de um fator de Cholesky da covariância. Com choques correlacionados, mover uma inovação sozinha não é um experimento que o modelo admite.

Os desvios devolvidos estão nas unidades reescaladas, vezes 100 quando se pedem porcentagens. Variáveis com \(\bar x_i \neq 0\) saem em desvio proporcional. Variáveis com \(\bar x_i = 0\) saem em desvio absoluto.

A covariância incondicional dos estados resolve a equação de Lyapunov discreta

\[
V_k = P V_k P^{\mathsf T} + Q \Sigma Q^{\mathsf T},
\]

e a do vetor completo é \(V = J V_k J^{\mathsf T}\), com \(J\) a matriz que empilha \(I\) sobre \(F\). Desvios-padrão, correlações e autocorrelações saem de \(V\) e de \(J P^j V_k J^{\mathsf T}\). São exatos para a solução linear. Uma simulação longa reproduz esses momentos. Ela não é a fonte deles.

A decomposição de variância atribui a cada choque a variância obtida com os demais desligados, em nível incondicional ou como variância do erro de previsão no horizonte \(h\),

\[
V_k^{(m)}(h) = \sum_{j=0}^{h-1} P^j Q \Sigma^{(m)} Q^{\mathsf T} (P^{\mathsf T})^j,
\]

com \(\Sigma^{(m)}\) a matriz que conserva só a entrada \((m,m)\). A participação é a razão entre a diagonal de \(J V_k^{(m)} J^{\mathsf T}\) e a soma sobre \(m\).

O teste de ordem de acurácia devolve as regras à função não linear em deslocamentos que encolhem. Uma solução de primeira ordem correta deixa resíduo de ordem \(\|\hat y\|^2\). Dividir o deslocamento por dois divide o resíduo por cerca de quatro. Um sistema montado com um bloco transposto deixa resíduo de ordem \(\|\hat y\|\), e a razão sai perto de dois. A razão é o teste. O tamanho de um resíduo isolado não é, porque um sistema errado pode ter resíduo pequeno num deslocamento pequeno e ainda assim ser de primeira ordem. Num modelo exatamente linear o resíduo está no arredondamento, e o relatório marca o caso em vez de inventar uma ordem a partir de ruído de máquina.

## 7. Segunda ordem

A regra, com os tensores contraídos nos estados, é

\begin{align*}
u_t &= F k_t + \tfrac12 g_{kk}[k_t, k_t] + \tfrac12 g_{ss}, \\
k_{t+1} &= P k_t + \tfrac12 h_{kk}[k_t, k_t] + \tfrac12 h_{ss} + Q\varepsilon_{t+1}.
\end{align*}

\(g_{kk}\) e \(h_{kk}\) são as derivadas segundas em relação aos estados. \(g_{ss}\) e \(h_{ss}\) são as correções de risco, o que a primeira ordem põe em zero por equivalência à certeza. Uma simulação de primeira ordem mantém a média dos estados no estado estacionário determinístico. Todo efeito de média que ela reporta é curvatura do mapa estático, sem o deslocamento das próprias regras. Com custo de ajustamento convexo esse deslocamento muda o sinal de um efeito de média, não só a magnitude. Ordenamentos de bem-estar dependem dele.

A hessiana de cada resíduo em relação a \(w = (\hat y_{t+1}, \hat y_t, \varepsilon)\) tem passo \(h = 10^{-4}\). A diferença segunda divide por \(h^2\). O arredondamento cresce como \(\varepsilon_{\mathrm{maq}}/h^2\) e o truncamento cai como \(h^2\), de modo que o equilíbrio fica perto de \(\varepsilon_{\mathrm{maq}}^{1/4}\). O passo \(10^{-6}\), adequado à primeira derivada, perde cerca de seis dígitos aqui. As derivadas mistas usam uma forma simétrica de seis pontos que reaproveita os deslocamentos de uma coordenada, ao custo de \(2m^2 + 2m\) avaliações em vez de \(4m^2\), com \(m = 2n + n_\varepsilon\). A derivada simbólica substitui essa hessiana quando o modelo foi declarado em texto.

Sejam \(f_1 = A\) e \(f_2 = -B\), já na escala de linha da seção 3, e \(H\) a hessiana na mesma escala. As derivadas do mapa composto em relação aos estados são

\[
M = \begin{pmatrix} P \\ FP \end{pmatrix}, \quad
N = \begin{pmatrix} I \\ F \end{pmatrix}, \quad
Z = \begin{pmatrix} M \\ N \\ 0 \end{pmatrix}.
\]

O termo conhecido da condição quadrática é \(K_e = Z^{\mathsf T} H_e Z\). Os blocos que multiplicam os tensores são

\begin{align*}
A_1 &= f_{1,k} + f_{1,u} F, \\
A_2 &= f_{1,u}, \\
A_3 &= f_{2,u},
\end{align*}

e \(W = (h_{kk}, g_{kk})\), de formato \((n, n_k, n_k)\), resolve a equação de Sylvester generalizada

\[
\bar A\, W + \bar B\, W\,(P \otimes P) = -K,
\]

com \(\bar A = [A_1 \mid A_3]\) e \(\bar B = [0 \mid A_2]\). A simetria de \(W\) nos dois índices de estado é imposta depois da solução. O resíduo da equação é reportado.

Enquanto \(n \cdot n_k^2\) não passa de 6000, o sistema é formado e fatorado. Acima disso um operador linear sem matriz, resolvido por LGMRES, substitui o produto de Kronecker denso, cujo armazenamento cresce com o quadrado dessa contagem.

As correções de risco não dependem de resolver de novo a Sylvester. Com \(\Sigma\) a covariância das inovações e

\[
L = \begin{pmatrix} Q \\ FQ \\ 0 \\ I \end{pmatrix}
\]

a derivada de \(w\) em relação às inovações,

\[
[A_1 \mid A_2 + A_3]
\begin{pmatrix} h_{ss} \\ g_{ss} \end{pmatrix}
  = -\Big( A_2\,(g_{kk} : Q\Sigma Q^{\mathsf T}) + H : L\Sigma L^{\mathsf T} \Big).
\]

Uma varredura de covariâncias reaproveita \(g_{kk}\), \(h_{kk}\) e \(H\).

No processo \(\log a' = \rho \log a + \varepsilon\), com \(\mathrm{Var}(\varepsilon) = \sigma^2\), tem-se \(E_t a' = a^{\rho} \exp(\sigma^2/2)\). Em desvio proporcional o deslocamento é \(\sigma^2/2\). A regra acrescenta \(h_{ss}/2\), logo \(h_{ss}\) sobre a tecnologia sai \(\sigma^2\). Onde a regra é exata nas coordenadas do modelo, como o capital no caso de depreciação total e utilidade logarítmica, a correção correspondente é zero. Uma correção fabricada nesse capital é um erro do bloco de risco.

### Poda e média ergódica

A simulação padrão é a poda de Kim, Kim, Schaumburg e Sims (2008). O termo quadrático é função da trajetória de primeira ordem apenas,

\begin{align*}
k^f_{t+1} &= P k^f_t + Q\varepsilon_{t+1}, \\
k^s_{t+1} &= P k^s_t + \tfrac12 h_{kk}[k^f_t, k^f_t] + \tfrac12 h_{ss}, \\
k_t &= k^f_t + k^s_t.
\end{align*}

Sem a poda o quadrático se alimenta de si mesmo. Uma amostra longa explode. Isso é o que a iteração de um mapa quadrático faz, e uma amostra não podada não sustenta um momento em horizonte em que o momento significaria alguma coisa.

A média ergódica sai em forma fechada da poda, não de uma média amostral. Com \(V_f\) a covariância de primeira ordem dos estados,

\[
E[k^s] = (I - P)^{-1}\Big(\tfrac12 h_{kk} : V_f + \tfrac12 h_{ss}\Big),
\]

e a média dos saltos carrega \(F\, E[k^s]\) mais os termos análogos em \(g_{kk}\) e \(g_{ss}\). O deslocamento é de ordem \(\sigma^2\) e o desvio-padrão da série é de ordem \(\sigma\). Separá-los por média amostral, num processo persistente, pede milhões de períodos. Uma amostra mais curta devolve ruído de Monte Carlo com sinal e magnitude plausíveis.

O teste de acurácia da segunda ordem desliga os termos de risco e lê só o bloco quadrático. O resíduo tem de cair como \(\|k\|^3\). Dividir o deslocamento por dois divide o resíduo por cerca de oito.

## 8. Terceira ordem, no que está implementado

A segunda ordem corrige a média das regras. A terceira corrige a inclinação em resposta ao risco, que é o que faz um prêmio de risco variar no tempo. A forma escrita no código seria

\begin{align*}
u_t &= Fk + \tfrac12 g_{kk}[k,k] + \tfrac12 g_{ss}
        + \tfrac16 g_{kkk}[k,k,k] + \tfrac12 g_{ssk}[k], \\
k_{t+1} &= Pk + \tfrac12 h_{kk}[k,k] + \tfrac12 h_{ss}
        + \tfrac16 h_{kkk}[k,k,k] + \tfrac12 h_{ssk}[k] + Q\varepsilon.
\end{align*}

Estão calculados \(g_{kkk}\) e \(h_{kkk}\). Eles resolvem a mesma Sylvester, com um fator de Kronecker a mais,

\[
\bar A\, W_3 + \bar B\, W_3\, (P \otimes P \otimes P) = -K_3,
\]

e a simetria nos três índices é imposta pela média das seis permutações. O termo conhecido \(K_3\) junta a hessiana contraída com as derivadas segundas do mapa e o tensor de terceira derivada contraído com \(Z\). A terceira derivada é uma diferença central da hessiana, com passo externo \(10^{-3}\) maior que o interno, porque diferenciar um objeto já obtido por diferenças pede um passo que domine o ruído desse objeto. Quando a hessiana é simbólica, essa diferença herda a exatidão dela. O tensor denso de terceira ordem tem ordem \(n m^3\). Num modelo de dezenas de variáveis ele não cabe como objeto simbólico, e o caminho desse tamanho é o exportador para o Dynare com `order=3`.

\(g_{ssk}\) e \(h_{ssk}\) são calculados a partir da curvatura da regra em \(\varepsilon\), antes de tomar a esperança. No AR(1) logarítmico, \(h_{ssk}[a,a] = \rho\sigma^2\), que é o termo cruzado de \((1+\hat a)^{\rho}\exp(\sigma^2/2)\). A solução de primeira ordem continua a somar \(Q\varepsilon\). O tensor em que a inovação é argumento de todos os monômios, \(y_t = g(y_{t-1}, u_t, \sigma)\), não é montado. O termo \(\sigma^2 k\) da regra de decisão é.

A simulação de terceira ordem é a poda em três camadas de Andreasen, Fernández-Villaverde e Rubio-Ramírez (2018). Cada camada é alimentada só pelas anteriores. O teste de acurácia, com o risco desligado, exige resíduo de ordem \(\|k\|^4\). Dividir o deslocamento por dois divide o resíduo por cerca de dezesseis.

## 9. Previsão perfeita

A transição determinística de um estado estacionário inicial a um terminal, em \(T\) períodos, usa a mesma \(f\), com \(\varepsilon = 0\) em toda data. As incógnitas são os \(n_u\) saltos em \(t = 0\) e o vetor completo em \(t = 1,\ldots,T\). Os \(n_k\) estados em \(t = 0\) vêm do estado inicial. Os \(n_u\) saltos em \(t = T\) são fixados no estado terminal. Fixar o vetor terminal inteiro sobredeterminaria o sistema em \(n_k\) equações. O sintoma é um Newton que deixa de convergir, com a aparência de um problema difícil.

O resíduo empilhado tem \(T n + n_u\) entradas. As primeiras \(Tn\) são \(f(y_{t+1}, y_t, 0)\) para \(t = 0,\ldots,T-1\). As últimas são a diferença entre os saltos terminais e os do estado estacionário de chegada. A jacobiana é bloco-bidiagonal e é montada esparsa. O passo padrão é diferença para a frente. A precisão da trajetória convergida é a tolerância do resíduo, padrão \(10^{-10}\), não a ordem da jacobiana. Diferença central fica disponível quando o resíduo é ruidoso.

Uma surpresa é uma mudança do estado terminal com parâmetros constantes. Uma trajetória anunciada é uma sequência de parâmetros, ou de contexto, de comprimento \(T+1\), conhecida desde \(t = 0\). As duas produzem impactos diferentes, e a diferença é o conteúdo do experimento. Parâmetros do período seguinte, quando a equação precisa deles, viajam no contexto, que também aceita uma trajetória.

## 10. Calibração por alvos

Dados parâmetros livres \(\theta\) e alvos \(m^\star\), a calibração resolve

\[
\frac{m(\theta) - m^\star}{|m^\star|} = 0
\]

em mínimos quadrados com limites, com denominador 1 quando o alvo é menor que \(10^{-12}\) em módulo. \(m(\theta)\) pode ser uma variável do estado estacionário ou uma função dela, como uma razão capital-produto. O estado estacionário é resolvido a cada avaliação, com partida quente no ponto anterior. A solução aceita é reavaliada no ponto devolvido, porque o último passo de tentativa do solver não é necessariamente o melhor. A tolerância padrão no erro relativo máximo é \(10^{-10}\).

A identificação é o valor singular da matriz de elasticidades \(\partial \log m_i / \partial \log \theta_j\), obtida por diferença com passo \(10^{-5}\). Posto deficiente significa uma direção de \(\theta\) que nenhum alvo vê. O solver para num ponto que depende da partida, e nenhuma tolerância corrige isso.

## 11. Condições de primeira ordem

Um problema recursivo declara o retorno do período \(u(y_t, k_t)\), controles \(y\), estados \(k\), restrições de igualdade \(h_j = 0\) e um parâmetro de desconto \(\beta\). O lagrangiano do período é

\[
\mathcal L_t = u(y_t, k_t) + \sum_j \lambda_{j,t}\, h_j(y_t, k_t, k_{t+1}).
\]

As condições devolvidas, como equações em texto prontas para o front end simbólico, são

\begin{align*}
\frac{\partial \mathcal L_t}{\partial y_t} &= 0, \\
\frac{\partial \mathcal L_t}{\partial k_{t+1}}
  + \beta\, E_t\!\left[\frac{\partial \mathcal L_{t+1}}{\partial k_{t+1}}\right] &= 0,
\end{align*}

mais as próprias restrições. O segundo termo da Euler é a derivada de \(\mathcal L_t\) em relação ao estado herdado, deslocada um período à frente. Se esse deslocamento produz um termo em \(t+2\), o problema não cabe em dois períodos e a função recusa, nomeando o estado. A derivada não é simplificada. Uma simplificação que dividisse o resíduo por uma variável deixaria de ser equivalente quando essa variável se aproximasse de zero.

Desigualdades, complementaridade e condições de transversalidade não são derivadas. O que sai é o que foi declarado.

## 12. Tradução da datação na exportação

O exportador só aceita modelo declarado em texto, porque uma função Python de resíduos não se descompila em equações. A parte que não é reimpressão mecânica é a datação, e é onde um porte manual erra.

Estoques predeterminados, os estados cuja linha de \(Q\) é nula, entram no Dynare como `predetermined_variables`. A variável sem defasagem é o estoque disponível em \(t\), e \(k_{t+1}\) escreve-se \(k(+1)\). No MacroModelling.jl o mesmo estoque é \(k[-1]\). Processos exógenos seguem a datação padrão do Dynare, em que a inovação se realiza em \(t\) e move a variável em \(t\). A equação é escrita um período atrás, \(\,a = \rho\, a(-1) + \varepsilon_a\,\). Escrevê-los com avanço produz um modelo que resolve e cujas correções de risco somem, pelo motivo da seção 1.

No Dynare os desvios-padrão vão no bloco `shocks`, inclusive as covariâncias fora da diagonal. No MacroModelling as inovações são normais padrão, e o desvio-padrão entra na equação como parâmetro `sig`. Omiti-lo produz variâncias unitárias.

## 13. Estimação, projeção, e o que continua de fora

A solução de primeira ordem é um espaço de estados linear. O filtro de Kalman lê a verossimilhança de séries em nível, centradas no estado estacionário dos parâmetros correntes. A moda da posteriori é um máximo numérico. As extrações são um Metropolis–Hastings de passeio aleatório. Um parâmetro que viola Blanchard–Kahn contribui com verossimilhança nula. Não há suavizador nem filtro de partículas.

A projeção coloca o mesmo resíduo numa grade de Chebyshev. As linhas da jacobiana sem salto futuro determinam o estado seguinte a cada nó de Gauss–Hermite. As outras são integradas. A política dos saltos é o polinômio. A grade tensorial para em três estados e a quadratura em dois choques.

Agentes heterogêneos no sentido de uma distribuição continuam fora. O mapa do que cada rotina não cobre está em `ROADMAP.md`.

Um equilíbrio de período calibrado numa matriz de contabilidade social não é uma instância de \(f\). Se um modelo desse tipo precisar de uma Euler ou de uma trajetória anunciada, o resíduo intertemporal é que viria para cá. A matriz permaneceria no outro pacote. A seção correspondente está no roadmap do `cgekit`.

## Referências

Andreasen, M. M., Fernández-Villaverde, J. e Rubio-Ramírez, J. F. (2018). The pruned state-space system for non-linear DSGE models: theory and empirical applications. *Review of Economic Studies*, 85(1), 1–49.

Blanchard, O. J. e Kahn, C. M. (1980). The solution of linear difference models under rational expectations. *Econometrica*, 48(5), 1305–1311.

Kim, J., Kim, S., Schaumburg, E. e Sims, C. A. (2008). Calculating and using second-order accurate solutions of discrete time dynamic equilibrium models. *Journal of Economic Dynamics and Control*, 32(11), 3397–3414.

Klein, P. (2000). Using the generalized Schur form to solve a multivariate linear rational expectations model. *Journal of Economic Dynamics and Control*, 24(10), 1405–1423.

Moler, C. B. e Stewart, G. W. (1973). An algorithm for generalized matrix eigenvalue problems. *SIAM Journal on Numerical Analysis*, 10(2), 241–256.

Schmitt-Grohé, S. e Uribe, M. (2004). Solving dynamic general equilibrium models using a second-order approximation to the policy function. *Journal of Economic Dynamics and Control*, 28(4), 755–775.
