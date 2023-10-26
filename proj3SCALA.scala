package scalation
package modeling
import scalation.*
import scalation.mathstat.{BASE_DIR, MatrixD, PlotM, VectorD}
import scalation.modeling.Regression
import scalation.modeling.RidgeRegression
import scalation.modeling.LassoRegression
import RidgeRegression.hp
import scala.collection.mutable.Set
import scala.runtime.ScalaRunTime.stringOf
import scalation.database.relation.Relation
import scalation.modeling.neuralnet.{NeuralNet_2L, NeuralNet_3L, NeuralNet_XL, Optimizer}
import scalation.modeling.ActivationFun.{f_aff, f_eLU, f_geLU, f_id, f_lreLU, f_reLU, f_sigmoid, f_softmax, f_tanh, logit_, sigmoid_}

@main def models (): Unit =
  val xr_fname = Array("Year","Violent","Property","Human Trafficking","Population","Index Crime Total","GDP")
  val temp = MatrixD.load("Crime-3.csv")
  val (x, y) = (temp.not(?,6), temp(?, 6)) // (data/input matrix, response column)

  val mu_x = x.mean // column-wise mean of x
  val mu_y = y.mean // mean of y
  val x_c = x - mu_x // centered x (column-wise)
  val y_c = y - mu_y // centered y

  val nullMod = new NullModel(y)
  nullMod.trainNtest ()()
  nullMod.summary()

  val yy = MatrixD(y_c).transpose
  var l = 10000000.0 // start with a small default value
  var l_best = l
  var sse = Double.MaxValue
  for i <- 0 to 10 do
    RidgeRegression.hp("lambda") = l
    val rrg = new RidgeRegression(x, y)
    val stats = rrg.crossValidate()
    val sse2 = stats(QoF.sse.ordinal).mean
    banner(s"RidgeRegession with lambda = ${rrg.lambda_} has sse = $sse2")
    if sse2 < sse then {
      sse = sse2;
      l_best = l
    }
    //          debug ("findLambda", showQofStatTable (stats))
    l *= 2
  end for

  RidgeRegression.hp("lambda") = 972.9
  banner("Ridge Regression")
  val mod = RidgeRegression.center(x, y, xr_fname) // create a ridge regression model (no intercept)
  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics

  banner("Optimize lambda")
  println(s"findLambda2 = ${mod.findLambda2(temp,y)}")

  banner("Cross-Validation")
  FitM.showQofStatTable(mod.crossValidate())

  println(s"x_fname = ${stringOf(xr_fname)}")

  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for RidgeRegression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for

  val stats = mod.crossValidate()
  FitM.showQofStatTable(stats)


  val mod2 = new LassoRegression(x, y) // create a Lasso regression model
  mod2.trainNtest()() // train and test the model
  println(mod2.summary()) // parameter/coefficient statistics

  println(s"best (lambda, sse) = ${mod2.findLambda}")

  banner("Cross-Validation")
  FitM.showQofStatTable(mod2.crossValidate())

  println(s"x_fname = ${stringOf(xr_fname)}")

  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod2.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for LassoRegression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for

  val stats2 = mod2.crossValidate()
  FitM.showQofStatTable(stats2)

end models








