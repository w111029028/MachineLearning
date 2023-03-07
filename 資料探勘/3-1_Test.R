male<-c(Amy=315,Bob=152,Kevin=337)
female<-c(Amy=346,Bob=126,Kevin=571)
v1<-c(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
v2<-c(1, 0.95, 0.95, 0.9, 0.85, 0.7, 0.65, 0.6, 0.55, 0.42)

tb<-rbind(male,female)

tb
chisq.test(tb)
plot(v1,v2)
cov(v1,v2)
cor(v1,v2)*sd(v1)*sd(v2)
cor.test(v1,v2)