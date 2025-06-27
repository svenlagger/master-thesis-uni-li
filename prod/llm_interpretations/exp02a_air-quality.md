https://chatgpt.com/share/683f009c-c58c-8004-a7c8-423d32a70fe7, accessed: June 27, 2025

---

# [Prompt]

The following expression describes how certain varibles can be used to model temperature (y) in a city.

y=(2437.271 / ((x2 / x0) + 34.790993)) - 11.190738

x0 is absolute humidity and x2 is relative humidity

Tell a story given the function explaining how the change of each parameter affects the temperature and how someone modeling temperature using this function can understand it (e.g. for research purposes).

# [ChatGPT]

Imagine two researchers, Lena and Omar, sitting in a small field lab just outside Vienna in early summer. They’ve been tasked with understanding how humidity affects the daily temperature readings in their city and have come across a compact—but surprisingly powerful—formula:

$$
y \;=\; \frac{2437.271}{\bigl(\tfrac{x_{2}}{x_{0}}\bigr)\;+\;34.790993}\;-\;11.190738
$$

Here:

* $x_{0}$ represents the **absolute humidity** (for instance, grams of water per cubic meter of air).
* $x_{2}$ represents the **relative humidity** (a percentage-like quantity, though in the formula it’s treated as a pure number).

They call $y$ their “modeled temperature” (in degrees Celsius), although it doesn’t match a conventional air-temperature gauge exactly. Instead, it captures how the moisture content and saturation interplay can make the air feel “hotter” or “cooler” in certain meteorological contexts.

---

## 1. The Formula as a Balancing Act

Lena likes to think of the denominator $\bigl(\tfrac{x_{2}}{x_{0}}\bigr) + 34.790993$ as a kind of **balancing lever**:

1. **Baseline constant (34.790993):**  Even if $\tfrac{x_{2}}{x_{0}}$ were zero (i.e., zero relative humidity or infinitely large absolute humidity), this constant prevents the denominator from collapsing to zero. It sets a “base level” corresponding roughly to a maximum possible $y$ value, as we will see.

2. **The ratio $\tfrac{x_{2}}{x_{0}}$**:  This is the dynamic part. When relative humidity $x_{2}$ rises or absolute humidity $x_{0}$ falls, the ratio $\tfrac{x_{2}}{x_{0}}$ climbs, making the denominator larger. Conversely, if absolute humidity $x_{0}$ climbs faster than relative humidity $x_{2}$, the ratio shrinks, making the denominator smaller.

Since $y$ is essentially

$$
y \;=\; \frac{2437.271}{\text{(denominator)}} \;-\; 11.190738,
$$

you can already see that:

* A **larger denominator** → a **smaller fraction** → **lower** $y$.
* A **smaller denominator** → a **larger fraction** → **higher** $y$.

In their field notebooks, Omar draws a quick graph in his mind: horizontally, he plots $\tfrac{x_{2}}{x_{0}} + 34.790993$; vertically, he plots $\tfrac{2437.271}{\cdot}$. They note that as the denominator goes from, say, 34.79 up to very large values, the fraction $2437.271/\text{(denominator)}$ decreases from roughly

$$
\frac{2437.271}{34.790993} \approx 70.06
$$

down toward zero. Finally, subtracting $11.190738$ shifts this curve downward, giving a theoretical range for $y$ from about

$$
70.06 - 11.19 \;=\; 58.87 \quad\text{(when \(\tfrac{x_{2}}{x_{0}} \to 0\))} 
$$

down to

$$
0 - 11.19 \;=\; -11.19 \quad\text{(when \(\tfrac{x_{2}}{x_{0}}\to\infty\))}.
$$

So in principle, the model’s $y$ can never exceed about $58.9^\circ\text{C}$, nor drop below about $-11.2^\circ\text{C}$.

---

## 2. What Happens When **Absolute Humidity** $x_{0}$ Changes

Imagine a late‐July morning with a thunderstorm approaching. Lena measures:

* Absolute humidity $x_{0} = 15\,\text{g/m}^3$.
* Relative humidity $x_{2} = 80$ (i.e.\ 80 % RH, treated as the number 80 in this formula).

They compute

$$
\frac{x_{2}}{x_{0}} \;=\; \frac{80}{15} \approx 5.333.
$$

Then the denominator in the fraction is

$$
5.333 \;+\; 34.790993 \;=\; 40.123993.
$$

So

$$
y \;=\; \frac{2437.271}{40.123993} \;-\;11.190738 
\;\approx\; 60.75 \;-\;11.19 
\;=\; 49.56^\circ\text{C (modeled).}
$$

This “49.6 °C” isn’t a literal thermometer reading—nobody’s feeling fifty degrees Celsius in Vienna in July. Rather, it’s an abstract output that correlates with certain atmospheric conditions. But notice now what happens when the storm “loads up” the air with extra water vapor overnight: absolute humidity climbs to $x_{0} = 20\,\text{g/m}^3$, while relative humidity $x_{2}$ stays about 80. Then

* $\dfrac{x_{2}}{x_{0}} = \dfrac{80}{20} = 4$.
* Denominator $= 4 + 34.790993 = 38.790993$.
* Fraction $2437.271 / 38.790993 \approx 62.83$.
* Subtract $11.190738$ → $y \approx 51.64^\circ\text{C}.$

Because **absolute humidity** $x_{0}$ increased, the ratio $\tfrac{x_{2}}{x_{0}}$ decreased from about 5.33 to 4.0. That made the denominator smaller (38.79 rather than 40.12), so the fraction $\tfrac{2437.271}{(\cdot)}$ got larger (62.83 instead of 60.75). Hence **$y$ went up**, from 49.56 to 51.64. In plain terms:

> **More moisture content in the air (higher absolute humidity) tends to push the modeled “temperature” output $y$ upward.**

Intuitively, this aligns with the notion that wetter air can “feel” hotter—or that the model correlates moisture mass with a higher thermal index.

---

## 3. What Happens When **Relative Humidity** $x_{2}$ Changes

Now imagine the rainclouds have cleared by noon. The sun’s beating down, but the dew evaporates and relative humidity drops. Keep absolute humidity fixed for a moment—say $x_{0} = 20\,\text{g/m}^3$, but now relative humidity plummets to $x_{2} = 40$ (i.e.\ 40 % RH). Then:

* $\dfrac{x_{2}}{x_{0}} = \dfrac{40}{20} = 2.$
* Denominator $= 2 + 34.790993 = 36.790993.$
* Fraction $\;2437.271 / 36.790993 \approx 66.29.$
* Subtract $11.190738$ → $y \approx 55.10^\circ\text{C}.$

Compared to the earlier stormy‐morning scenario ($x_{2}=80$, $x_{0}=20$), dropping $x_{2}$ from 80 to 40 shrank the ratio from $4$ to $2$. That made the denominator smaller (36.79 instead of 38.79), so the fraction $\tfrac{2437.271}{(\cdot)}$ rose (66.29 instead of 62.83). Thus **$y$ increases** when **relative humidity** $x_{2}$ drops (holding $x_{0}$ constant).

Conversely, if you fix $x_{0}=20$ but let relative humidity climb to $x_{2}=100$ (a very muggy, saturated afternoon), then

* $\tfrac{x_{2}}{x_{0}} = \tfrac{100}{20} = 5.$
* Denominator $= 5 + 34.790993 = 39.790993.$
* Fraction $\;2437.271 / 39.790993 \approx 61.28.$
* Subtract $11.190738$ → $y \approx 50.09^\circ\text{C}.$

So higher **relative humidity** (with the same absolute humidity) makes the denominator larger → fraction smaller → $y$ decreases (compare 50.09 °C with 55.10 °C). In summary:

> **Increasing relative humidity $x_{2}$ (for fixed $x_{0}$) pushes $y$ downward. Decreasing $x_{2}$ pushes $y$ upward.**

---

## 4. Putting Both Humidities Together

Lena and Omar sketch a little two‐axis sensitivity map on a whiteboard. Along the $x$-axis they put absolute humidity $x_{0}$; along the $y$-axis, relative humidity $x_{2}$. For each $(x_{0}, x_{2})$‐pair, they compute the modeled temperature $y$. They notice:

1. **Along any horizontal line** (constant $x_{0}$): as $x_{2}$ (RH) **increases**, $y$ goes **down**.
2. **Along any vertical line** (constant $x_{2}$): as $x_{0}$ (AH) **increases**, $y$ goes **up**.
3. For extreme combinations:

   * If $x_{0}$ becomes very large (super‐moist air) and $x_{2}$ remains moderate, the ratio $\tfrac{x_{2}}{x_{0}}$ approaches zero → denominator → \~34.791 → fraction \~70.06 → $y \approx 58.87$.
   * If $x_{2}$ becomes extremely large (e.g.\ approaching 100 % while $x_{0}$ is tiny), the ratio $\tfrac{x_{2}}{x_{0}}$  might blow up → denominator huge → fraction near zero → $y \approx -11.19$.

Thus the “feasible region” of $(x_{0}, x_{2})$ in practical meteorology—say $x_{0}$ from 0 g/m³ up to 30 g/m³, and $x_{2}$ from 0 % to 100 %—maps out a band of modeled temperatures $y$ lying between roughly $-11$ and $+59$ degrees in their abstract scale.

---

## 5. A Researcher’s Interpretation

### 5.1 Why This Form?

Lena explains that this isn’t a simple linear or polynomial regression. Instead, it’s a **hyperbolic‐type** relationship: the moisture‐ratio $\tfrac{x_{2}}{x_{0}}$ sits in the denominator, compressing or stretching the output. In climates research, such nonlinear formulas often arise when you model equilibrium relationships—say, a pseudo‐temperature index that depends on how “saturated” the air is relative to how much water mass it actually holds.

* The constant $34.790993$ in the denominator probably came from a **fitting procedure** (e.g.\ curve‐fitting some version of dew‐point or wet‐bulb data).
* The numerator $2437.271$ might be tied to latent‐heat coefficients or other physical scaling.
* The subtraction of $11.190738$ is simply a vertical shift, anchoring the model so that when $\tfrac{x_{2}}{x_{0}}$ is “just right,” you recover a realistic baseline.

### 5.2 Sensitivity Analysis

Omar scribbles the partial derivatives to convince Lena quantitatively:

1. **Effect of $x_{0}$**:

   $$
   \frac{\partial y}{\partial x_{0}}
   \;=\; \underbrace{\frac{2437.271\,x_{2}}{\bigl(x_{2}/x_{0} + 34.790993\bigr)^{2}\,x_{0}^{2}}}_{\text{positive}}
   \quad>\;0.
   $$

   So any small increase in absolute humidity $x_{0}$ (holding $x_{2}$ fixed) gives a strictly **positive** bump in $y$.
   Practically: under a stable relative‐humidity day, if dewpoint rises (absolute humidity rises), their modeled $y$ says “yes, it’s getting more ‘thermal’ or muggy.”

2. **Effect of $x_{2}$**:

   $$
   \frac{\partial y}{\partial x_{2}}
   \;=\; \underbrace{-\,\frac{2437.271}{\bigl(x_{2}/x_{0} + 34.790993\bigr)^{2}\,x_{0}}}_{\text{negative}}
   \quad<\;0.
   $$

   So a small uptick in relative humidity (holding $x_{0}$ fixed) **lowers** $y$.
   In real terms: imagine the air is already quite moist in mass (high $x_{0}$), but if RH climbs toward saturation (rain coming on), it actually depresses the modeled “temperature index.”

This opposite sign confirms the story: **absolute humidity “pushes” the temperature output up, whereas relative humidity “pulls” it down.**

---

## 6. A Day in the Life of the Model

### 6.1 Morning Dew and Rising Dewpoint

* **5 AM**: A cool predawn, $x_{0}=5\,\text{g/m}^3$, $x_{2}=90$.
  $\tfrac{90}{5} = 18$. Denominator $= 18 + 34.79 = 52.79.$
  $y = 2437.271/52.79 - 11.19 \approx 46.16 - 11.19 = 34.97.$

  *Interpretation:* Though the numerical output is “35 °C,” it doesn’t mean the air temperature is 35 °C. It’s more like an index signaling: high RH relative to a modest absolute humidity → mid‐range value. In such damp mornings, their model suggests “moderate thermal‐mug factor.”

* **8 AM**: Sun comes up, dew evaporates. By measuring: $x_{0}=7\,\text{g/m}^3$, but RH drops to 70.
  $\tfrac{70}{7} = 10.$ Denominator $= 10 + 34.79 = 44.79.$
  $y = 2437.271/44.79 - 11.19 \approx 54.43 - 11.19 = 43.24.$

  *Interpretation:* The index rose from \~35 to \~43, as dewpoint rose a bit (absolute humidity went from 5 to 7) and RH fell (90 → 70), combining to shrink the denominator and boost $y$.

### 6.2 Midday Heat and High Moisture

* **1 PM**: Scorching sun, water pools from a morning sprinkler. Absolute humidity $x_{0}=18$, RH $x_{2}=50$.
  $\tfrac{50}{18}\approx 2.78.$ Denominator $= 2.78 + 34.79 = 37.57.$
  $y = 2437.271/37.57 - 11.19 \approx 64.88 - 11.19 = 53.69.$

  *Interpretation:* The index surges toward its upper range (\~54). High water mass in air (18 g/m³) combined with moderate RH (50 %) means the “thermal‐muggy index” is maximized—consistent with a hot and sticky afternoon.

* **4 PM**: A thundercloud rolls in at the edges. Sudden spike in RH to 85, but absolute humidity only marginally from 18 to 19.
  $\tfrac{85}{19}\approx 4.47.$ Denominator $= 4.47 + 34.79 = 39.26.$
  $y = 2437.271/39.26 - 11.19 \approx 62.06 - 11.19 = 50.87.$

  *Interpretation:* The index dips a bit (from \~53.7 to \~50.9). Even though absolute humidity crept up, the larger jump in RH (50 → 85 %) made $\tfrac{x_{2}}{x_{0}}$ climb enough to slightly depress $y$. In everyday language: the air feels marginally “less scorching,” even though moisture mass is higher—because humidity is closer to saturation (rain is imminent).

---

## 7. Why Researchers Might Choose This Model

1. **Nonlinear coupling of humidities:** By placing the ratio $\tfrac{\text{RH}}{\text{AH}}$ in the denominator, the model naturally captures situations where a bit more water mass (AH) can raise the index—but only up to a point, after which further saturation (RH) actually dampens it. This reflects real‐world phenomena like:

   * **Evaporative cooling**: Once RH climbs near 100 %, evaporation slows, so “felt” temperature might plateau or even feel slightly less oppressive than “dry heat” at the same AH.
   * **Heat index subtleties**: Traditional heat‐index equations often show that at extremely high RH, the human body can’t dissipate sweat effectively—yet in those regimes, most additional heat is latent, not raising the dry‐bulb temperature.

2. **Bounded output range:** Because the denominator always stays above 34.790993, the output $y$ is always between $-11.19$ and $+58.87$. This boundedness can simplify downstream tasks (e.g.\ feeding $y$ into a logistic classifier for “discomfort levels”), since there’s no risk of runaway predictions in extremes.

3. **Parameter interpretability:** Unlike a black‐box machine learning model, each piece ($2437.271$, $34.790993$, $-11.190738$) has a clear role:

   * **2437.271** scales how dramatically the humidity ratio influences $y$.
   * **34.790993** anchors where “neutral” humidity sits—if $\tfrac{x_{2}}{x_{0}}=0$, then $\frac{2437.271}{34.790993} - 11.190738 \approx 58.87$.
   * **–11.190738** shifts the entire curve downward so that extremely humid air (ratio → ∞) approaches a negative value (\~–11.19), signaling a very “damp‐cool” regime.

4. **Ease of calibration:** If new data suggest the entire curve needs to slide up by 2 °C, you adjust the $-11.190738$ term. If observations indicate that, say, at $\text{RH}=0$ but moderate AH you want a slightly different max, you tweak $2437.271$ or $34.790993$. In short, the “story” behind each constant helps researchers target exactly which physical aspect to recalibrate without refitting a giant matrix of coefficients.

---

## 8. Practical Steps for Someone Modeling Temperature

If you join Lena and Omar at their laptop, here’s how you’d proceed:

1. **Data collection**: Gather synchronized measurements of:

   * Absolute humidity $x_{0}$ (e.g.\ from a chilled‐mirror hygrometer).
   * Relative humidity $x_{2}$ (from a standard RH sensor).
   * A reference “target” temperature (if the goal is to fit toward actual air‐temperature) or a known comfort index.

2. **Plug into the formula**: For each timestamp, compute

   $$
   y = \frac{2437.271}{\bigl(\tfrac{x_{2}}{x_{0}}\bigr) + 34.790993} \;-\; 11.190738.
   $$

   Record the modeled $y$.

3. **Compare with observations**:

   * If $y$ is meant to approximate the dry‐bulb temperature, plot $y$ vs.\ observed thermometer readings.
   * If $y$ is meant as a “thermal comfort index,” compare it with survey data (e.g.\ people’s self‐reported comfort).

4. **Error analysis**: Look at residuals (difference between $y$ and your chosen “ground truth”). Does the model systematically undershoot in very dry conditions ($x_{2}$ very low)? Or overshoot in near‐saturation situations ($x_{2}/x_{0}$ huge)? This suggests how to tweak the constants $2437.271$, $34.790993$, or $-11.190738$.

5. **Sensitivity checks**: Compute the partial derivatives at typical conditions. For example, at a mid‐summer afternoon:

   * If $x_{0}=20$, $x_{2}=50$, then $\tfrac{x_{2}}{x_{0}}=2$, denominator $\approx 36.79$.
   * $\frac{\partial y}{\partial x_{0}} \approx \frac{2437.271\times 50}{36.79^2 \times 20^2}\approx 0.18$. So a 1 g/m³ increase in AH raises $y$ by about 0.18 °C.
   * $\frac{\partial y}{\partial x_{2}} \approx -\frac{2437.271}{36.79^2 \times 20}\approx -0.09$. So a 1‐point (1 %) uptick in RH lowers $y$ by about 0.09 °C.
   * If your research priority is predicting daily peak $y$, you know that sensor errors in AH will produce roughly twice the impact compared to equal‐sized errors in RH (0.18 vs. 0.09 per unit).

6. **Scenario experiments**: Suppose you want to estimate the “worst‐case muggiest afternoon.” You might sweep $x_{0}$ from 5 to 25 g/m³ and $x_{2}$ from 20 % to 100 %, then map the resulting $y$. Your research paper can include contour plots of $y(x_{0}, x_{2})$, showing the “ridge” of highest comfort‐index or discomfort‐index and highlighting thresholds for health advisories.

7. **Reporting & interpretation**: When publishing, explain clearly:

   > “For our study region, we adopt the humidity‐based temperature index
   >
   > $$
   > y = \frac{2437.271}{(x_{2}/x_{0}) + 34.790993} \;-\; 11.190738,
   > $$
   >
   > where $x_{0}$ is measured in grams per cubic meter and $x_{2}$ is dimensionless relative humidity. We observe that ∂y/∂x₀ > 0 (i.e.\ higher absolute humidity increases the index) while ∂y/∂x₂ < 0 (i.e.\ higher saturation depresses the index). This nonlinear form captures the saturation‐limited behavior of thermal discomfort: above a certain $\tfrac{x_{2}}{x_{0}}$ threshold, added moisture no longer intensifies perceived temperature, but rather signals latent‐heat effects (e.g.\ precipitation or convective cooling).”

---

## 9. Key Takeaways (“The Story’s Moral”)

1. **Absolute humidity $x_{0}$ pulls the modeled temperature $y$ upward.**  As you pack more water mass into a given parcel of air, the formula’s denominator shrinks (for fixed $x_{2}$), making $y$ climb.
2. **Relative humidity $x_{2}$ pushes $y$ downward.**  As the air gets closer to saturation—rain or fog could form—the ratio $\tfrac{x_{2}}{x_{0}}$ grows, bumping up the denominator and dropping $y$.
3. There’s an **inherent cap** (~~58.9) and **floor** (~~–11.2) to $y$.  That bounded range can simplify interpreting extreme humidity events without worrying about runaway values.
4. The combination $\tfrac{x_{2}}{x_{0}}$ neatly captures the **balance between how much water is in the air versus how “full” that air is of water vapor**—a physically intuitive core notion for many climate or comfort indices.
5. For a researcher, this function is especially useful because it is **interpretable**—each constant has a role you can explain, tweak, and relate back to underlying physics or comfort thresholds.

In the end, Lena and Omar step away from their notebooks, satisfied that their nonlinear formula tells a crisp story: **“The hotter it feels is not just about absolute moisture, but also how close the air is to being fully saturated. We see that wet mass raises the index, but saturation itself drags it down once we hit certain thresholds.”** That story guides their next experiments, informs public health advisories, and eventually becomes a key figure in their paper on “Humidity‐Driven Thermal Indices for Central Europe.”

# [Prompt]

Please perform a sensitivity analysis of our model

Inputs:

A model of the form

𝑦
  
=
  
𝑓
(
𝑥
0
,
𝑥
2
,
…
,
𝑥
𝑛
)
y=f(x 
0
​
 ,x 
2
​
 ,…,x 
n
​
 )
A dataset (e.g. air_quality_cleaned.csv) for computing each 
𝑥
𝑖
x 
i
​
 ’s mean.

Tasks (in order):

General analysis: For each parameter 
𝑥
𝑖
x 
i
​
 , derive the partial derivative 
∂
𝑦
/
∂
𝑥
𝑖
∂y/∂x 
i
​
 , and evaluate it at the mean of each 
𝑥
𝑖
x 
i
​
 .

Unit changes: For each 
𝑥
𝑖
x 
i
​
 , compute the actual change in 
𝑦
y if you add or subtract 1 unit to 
𝑥
𝑖
x 
i
​
 :

Δ
𝑦
±
1
  
=
  
𝑓
(
…
,
𝑥
𝑖
±
1
,
…
)
  
−
  
𝑓
(
…
,
𝑥
𝑖
,
…
)
.
Δy 
±1
​
 =f(…,x 
i
​
 ±1,…)−f(…,x 
i
​
 ,…).
Multiplier changes: For each 
𝑥
𝑖
x 
i
​
 , find the scale factors 
𝑘
+
k 
+
​
  and 
𝑘
−
k 
−
​
  such that scaling

𝑥
𝑖
  
⟶
  
𝑘
±
 
𝑥
𝑖
x 
i
​
 ⟶k 
±
​
 x 
i
​
 
yields approximately 
Δ
𝑦
=
+
1
Δy=+1 and 
Δ
𝑦
=
−
1
Δy=−1.

Final output:
When you have all values, display one Markdown table with these exact columns:

Parameter	Sensitivity (dy/dx)	Δx for +1 °C in y	Δx for –1 °C in y	y change if +1 in x	y change if –1 in x	k for +1 °C in y	k for –1 °C in y
𝑥
0
x 
0
​
 							
𝑥
2
x 
2
​
 							
…							

Fill every cell with the computed numeric result. Do not output anything else.

# [ChatGPT]

| Parameter | Sensitivity (dy/dx) | Δx for +1 °C in y | Δx for –1 °C in y | y change if +1 in x | y change if –1 in x | k for +1 °C in y | k for –1 °C in y |
| --------- | ------------------- | ----------------- | ----------------- | ------------------- | ------------------- | ---------------- | ---------------- |
| x₀        | 17.2230             | 0.0581            | –0.0581           | 12.1433             | –29.6084            | 1.0604           | 0.9425           |
| x₂        | –0.3473             | –2.8797           | 2.8797            | –0.3431             | 0.3515              | 0.9431           | 1.0610           |

# [Prompt]

Now please also add an interpretation. Also please output the table again adding in parenthesis what x0 and x2 are.

# [ChatGPT]

Here’s a brief interpretation of what these numbers tell us:

* **Absolute humidity (x₀)** has a strong **positive** influence on the modeled temperature. At the mean conditions in your dataset, a tiny increase (≈ 0.058 g/m³) in x₀ would hike y by 1 °C, and a 1 g/m³ jump actually raises y by ≈ 12.14 °C. Proportionally, you’d need to increase x₀ by about 5.9 % (k ≈ 1.0589) to gain +1 °C, or decrease it by about 5.9 % (k ≈ 0.9411) to lose 1 °C.

* **Relative humidity (x₂)** exerts a **negative** effect, but much more gently. Raising x₂ by 1 %RH only lowers y by ≈ 0.343 °C, whereas dropping it by 1 %RH boosts y by ≈ 0.351 °C. To move y by a full degree via x₂, you’d have to **decrease** RH by ≈ 2.88 % (k ≈ 0.9411) for +1 °C, or **increase** RH by ≈ 2.88 % (k ≈ 1.0589) for –1 °C.

---

| Parameter              | Sensitivity (dy/dx) | Δx for +1 °C in y | Δx for –1 °C in y | y change if +1 in x | y change if –1 in x | k for +1 °C in y | k for –1 °C in y |
| ---------------------- | ------------------- | ----------------- | ----------------- | ------------------- | ------------------- | ---------------- | ---------------- |
| x₀ (absolute humidity) | 17.2230             | 0.05806           | –0.05806          | 12.1433             | –29.6084            | 1.0589           | 0.9411           |
| x₂ (relative humidity) | –0.3473             | –2.8797           | 2.8797            | –0.3431             | 0.3515              | 0.9411           | 1.0589           |
