"""Replacement tasks. Correct answer is first in each authoring list;
the release builder redistributes its visible position across A-D.
These are assistant-authored specifications, not independent human labels.
"""
SCIENCE = {
 'logic_117': (
  'A magma sample is observed to contain molten silicate material with suspended solid mineral crystals. The revised task concerns this sample, rather than the general composition of rocks. Which statement reports its observed physical state and contents?',
  ['The sample contains molten material and solid mineral crystals.', 'The sample is entirely solid rock.', 'The sample contains only gas.', 'The sample contains no solid mineral crystals.'],
  'The observation establishes both molten material and suspended solid crystals; the other three descriptions contradict it.'),
 'logic_219': (
  'After a ball is thrown diagonally through still air, its horizontal speed decreases before it touches anything. Use a model with gravity acting vertically and air resistance opposing motion; neglect other forces. Which force accounts for the decrease in horizontal speed?',
  ['Air resistance opposing the horizontal motion.', 'Gravity acting vertically downward.', 'A horizontal push from the ground.', 'A continued horizontal push from the thrower.'],
  'The stated model gives gravity no horizontal component and excludes contact forces during flight. Air resistance supplies the opposing horizontal component.'),
 'logic_299': (
  'All hay fields are bare today. After planting, each field requires two full years before its first hay harvest. No other hay is available. Which of the following planting plans can yield a hay harvest from these fields within the next 12 months?',
  ['None of these planting plans can yield a harvest within 12 months.', 'Plant all fields today.', 'Plant one field today and the others six months later.', 'Plant a different field at each one-year interval.'],
  'Even the earliest possible planting matures after 24 months, which exceeds the 12-month window.'),
 'logic_39': (
  'Equal initial volumes of hydraulic oil and pneumatic air are subjected to the same pressure increase. The oil volume decreases by 0.001 mL and the air volume by 5 mL. Which sample has the larger measured volume decrease?',
  ['The pneumatic air.', 'The hydraulic oil.', 'Both samples have the same decrease.', 'Neither sample decreases in volume.'],
  '5 mL exceeds 0.001 mL; this asks for a measured comparison rather than the classification of hydraulic and pneumatic systems.'),
 'logic_92': (
  'Salt dissolves completely and uniformly in one jar of water. Insoluble pepper particles remain suspended or settle in a second jar. Which classification distinguishes these two jars? A solution is a homogeneous mixture of dissolved substances.',
  ['Salt water is a solution; pepper water is a mixture but not a solution.', 'Both jars are pure substances.', 'Pepper water is a solution; salt water is a mixture but not a solution.', 'Neither jar is a mixture.'],
  'Dissolved, uniform salt water meets the supplied solution definition; undissolved pepper does not. Both remain mixtures, so the original broader classification does not answer the revised distinction.'),
 'logic_221': (
  'Salt dissolves completely and uniformly in water. The task now asks for the name of the entire homogeneous mixture, rather than the separate roles of salt and water. What is the entire mixture called?',
  ['A solution.', 'A solute.', 'A solvent.', 'A pure element.'],
  'The mixture as a whole is a solution; solute and solvent name its constituents.'),
 'logic_21': (
  'The task now concerns only the hydrogen-producing cathode half-reaction in alkaline water electrolysis, rather than the overall cell reaction. Which equation has electrons on the reactant side, produces hydrogen, and balances both atoms and charge?',
  ['2H2O(l) + 2e^- -> H2(g) + 2OH^-(aq)', '2H2O(l) -> 2H2(g) + O2(g)', '4OH^-(aq) -> O2(g) + 2H2O(l) + 4e^-', '2H2O(l) + 2e^- -> H2(g) + 2OH^+(aq)'],
  'The target balances H=4, O=2, and charge=-2 on each side. The overall equation is not a cathode half-reaction, the oxygen equation is anodic, and the last option has unequal charge.'),
 'logic_267': (
  'A survey documents the effects of one tornado in a treeless inland grassland. It records extensive animal-habitat loss, no fallen trees, no coastal storm surge, and no change in mosquito breeding sites. Which effect was documented in this surveyed event?',
  ['Extensive loss of animal habitat.', 'Many trees fell.', 'A coastal storm surge destroyed turtle nests.', 'Mosquito breeding sites increased.'],
  'Only habitat loss appears in the survey; the other three effects are explicitly absent. The task asks what occurred, rather than which effect is not adverse.'),
 'logic_266': (
  'For this observation period, a cactus has no access to groundwater or rain. It remains hydrated while measurements show a decline in the water stored in its stem. Which water source is directly supported by these observations?',
  ['Water stored in the cactus stem.', 'New groundwater absorbed during the observation period.', 'Rain absorbed during the observation period.', 'Water supplied by nearby animals during the observation period.'],
  'The stored-water decrease supplies the supported source; rain and groundwater are excluded and animal supply is not observed.'),
 'logic_57': (
  'A demonstration connects a coil to a small light bulb. Moving a magnet through the coil induces a current, and the bulb lights when that current flows. The revised task asks for the role of the bulb, rather than the role of the generator. What does the bulb do in this demonstration?',
  ['It indicates that electric current is flowing.', 'It creates the motion of the magnet.', 'It supplies the permanent magnetic field.', 'It generates the induced current in the coil.'],
  'The coil and moving magnet induce current; the observed illumination makes the bulb an indicator, not its generator.'),
 'logic_65': (
  'Before fertilization, homologous chromosomes in a cell undergoing meiosis exchange corresponding DNA segments. Which process explains the resulting new combinations of alleles in the gametes?',
  ['Crossing over between homologous chromosomes.', 'Fusion of two parental gametes at fertilization.', 'Cloning an unchanged chromosome set.', 'Copying DNA without any exchange between chromosomes.'],
  'The specified exchange during meiosis is crossing over. Fertilization has not yet occurred and the unchanged-copy distractors cannot explain the exchange.'),
 'logic_8': (
  'A farm census counts 100 chickens immediately before hawks are removed and 125 chickens one month later. No counts of mice, rats, or other predators are available. Which conclusion follows directly from these two censuses?',
  ['The chicken population increased over the month.', 'The chicken population decreased over the month.', 'The mouse population doubled over the month.', 'The rat population remained unchanged over the month.'],
  '125 exceeds 100. The censuses support no numerical conclusion about rodents and do not by themselves establish the cause of the increase.'),
 'logic_107': (
  'During a specified monitoring period, a tapeworm obtains nutrients and grows inside a dog. Veterinary measurements show no detectable illness or nutritional decline in this dog during that period. Which statement reports these observations without generalizing to all tapeworm infections?',
  ['The tapeworm benefits, and no harm to this dog is detected during the monitoring period.', 'Both the tapeworm and this dog show measurable nutritional decline.', 'This dog becomes observably ill during the monitoring period.', 'The tapeworm receives no nutrients and does not grow.'],
  'Growth and nutrient intake establish the worm benefit; the measured absence of detectable harm is bounded to this dog and period. This is not a claim that tapeworm infections generally cause no harm.'),
 'logic_16': (
  'A fossil image contains ten fully preserved, outlined skin scales and a calibrated length scale bar. The task is to estimate the arithmetic mean area of those ten scales. Which procedure directly computes that quantity?',
  ['Use the scale bar to measure each scale area and average the ten areas.', 'Measure each scale perimeter and average the ten perimeters.', 'Count the ten scales and report the count as their mean area.', 'Ignore the scale bar and report pixel lengths as square centimeters.'],
  'A calibrated area measurement for each scale followed by sum/10 yields the requested mean area. Perimeter, count, and uncalibrated lengths have incompatible dimensions.'),
 'logic_209': (
  'Two identical twins have the same measured genetic markers and blood groups but different fingerprints. Known fingerprint records from both twins are available, along with a clear fingerprint left on a glass. Which comparison can distinguish the twins using the available evidence?',
  ['Compare the print on the glass with both known fingerprint records.', 'Compare only the measured genetic markers that are identical.', 'Compare only the blood groups that are identical.', 'Infer the identity from the fact that the twins have the same age.'],
  'Fingerprints are specified as distinct and available; the other listed features are identical and cannot distinguish these twins.'),
 'logic_138': (
  'The maximum supported load of a proposed bridge has already been verified. The revised engineering task is to evaluate movement at its expansion joints during a specified design earthquake. Which reported quantity directly measures that movement?',
  ['Maximum side-to-side displacement at the expansion joints during the design earthquake.', 'Maximum static load the bridge can support.', 'Number of vehicles painted blue on the bridge.', 'Total length of the approach road.'],
  'Displacement measures the requested joint movement. Static capacity addresses the earlier task and the other quantities are unrelated.'),
 'logic_178': (
  'The compound in the revised task is magnesium chloride, rather than magnesium bromide. Magnesium forms Mg2+ ions and chloride forms Cl^- ions. Which formula gives an electrically neutral formula unit?',
  ['MgCl2', 'MgCl', 'Mg2Cl', 'MgBr2'],
  'One +2 magnesium ion requires two -1 chloride ions; the bromide formula uses the wrong anion.'),
 'logic_102': (
  'In the observed surgical step, a defocused laser heats tissue around a small bleeding vessel. The vessel closes by coagulation; no incision is made. Which action is the laser performing in this step?',
  ['Sealing the bleeding vessel by coagulation.', 'Making a fine incision through the vessel.', 'Removing the entire organ containing the vessel.', 'Mechanically tying a thread around the vessel.'],
  'The observation specifies heat-induced coagulation and closure, excludes an incision, and supplies no mechanical tying or organ removal.'),
 'logic_77': (
  'A permanent pond has ample water throughout the season. Tadpoles normally finish metamorphosis on day 25. Predators arrive on day 20 and eat tadpoles but not the transformed frogs. In this scenario, why would completing metamorphosis on day 15 help the tadpoles survive?',
  ['It finishes metamorphosis before the tadpole-eating predators arrive.', 'It compensates for the pond running out of water before day 15.', 'It postpones metamorphosis until after the predators arrive.', 'It increases the number of days spent as tadpoles while predators are present.'],
  '15 < 20 < 25. Earlier completion leaves no tadpole stage at predator arrival; water shortage is explicitly absent.'),
 'logic_26': (
  'A report summarizes 30 years of records showing that flash-flood risk in a region is usually higher in one season than another. It is describing the long-term seasonal pattern, not issuing a warning for a particular day. How should this reported pattern be classified?',
  ['A seasonal climate pattern of flash-flood risk.', 'A weather observation for one specified day.', 'A prediction of the exact time of every future flood.', 'Evidence that floods cannot occur outside the higher-risk season.'],
  'The multi-decade seasonal summary describes climate; neither exact event timing nor impossibility in another season follows.'),
 'logic_123': (
  'The revised task asks why tropical cyclones rarely originate directly at the equator, rather than why the tropics receive strong solar heating. Which feature makes it difficult for equatorial storms to acquire organized rotation?',
  ['The Coriolis deflection of horizontal winds is zero at the equator.', 'The Coriolis deflection is strongest at the equator.', 'Equatorial ocean water is always frozen.', 'Solar radiation never reaches the equator.'],
  'The Coriolis parameter is 2*Omega*sin(latitude), which is zero at latitude zero. This inhibits rotation; the wording allows exceptional near-equatorial storms rather than asserting absolute impossibility.'),
 'logic_275': (
  'For the revised task, compare a measured average car speed of 339 m/s with a supplied reference speed of sound of 343 m/s. Do not infer the car\'s instantaneous peak speed or a historical record. Which numerical comparison is correct?',
  ['The measured average speed is less than the reference speed.', 'The measured average speed is greater than the reference speed.', 'The measured average speed equals the reference speed.', 'The speeds cannot be compared because their units differ.'],
  '339 < 343 and both have units m/s. An average-to-reference comparison does not determine a peak-speed record.'),
 'logic_152': (
  'At a coastal cliff, repeated wave impacts detach and carry away pieces of exposed bedrock, leaving a flat wave-cut platform at the cliff foot. No lava deposits or sediment buildup are present. Which process formed the observed platform?',
  ['Erosion of bedrock by waves.', 'Buildup of cooled lava.', 'Deposition of river sediment.', 'Construction of a coral reef on the rock.'],
  'Observed removal of bedrock by waves is marine erosion; the depositional mechanisms are absent.'),
 'logic_73': (
  'Between two surveys, a river bed rises from deposited sediment and the banks move outward. Compare channel depth and width at the same water-surface elevation in both surveys. How has the channel changed?',
  ['It has become shallower and wider.', 'It has become deeper and wider.', 'It has become shallower and narrower.', 'It has become deeper and narrower.'],
  'A higher bed decreases depth at the fixed surface elevation; outward bank movement increases width.'),
 'logic_194': (
  'Rainfall is being compared at several locations using identical rain gauges. One gauge is under a roof and the others have unobstructed exposure. The revised task is to remove this exposure difference. Which change directly addresses it?',
  ['Move the sheltered gauge to an unobstructed location comparable to the others.', 'Leave the sheltered gauge under the roof because the gauges are identical.', 'Put a lid on every gauge during rain.', 'Paint each gauge a different color.'],
  'Identical instruments do not remove shelter bias; comparable unobstructed exposure addresses the stated difference.'),
 'logic_207': (
  'A bag is warmer than the chocolate chips touching its inner surface. For this step, consider heat transfer across the bag-to-chocolate contact boundary, rather than across the boundary of the entire surroundings. In which direction does heat flow at that contact?',
  ['From the warmer bag to the cooler chocolate chips.', 'From the cooler chocolate chips to the warmer bag.', 'No heat can flow between touching objects.', 'Only from the surrounding air directly to the chocolate, without passing through the bag.'],
  'At the specified contact, passive heat transfer is from the warmer bag to the cooler chocolate. The earlier broad surroundings answer does not identify this boundary.'),
 'logic_214': (
  'The revised experiment asks whether stove type affects the fraction of popcorn kernels that pop. Which design compares an electric stove with a gas stove while controlling popcorn type, number of kernels, pot, and heating duration?',
  ['Use equal counts of the same popcorn type on each stove with matched pots and durations, and repeat the trials.', 'Change popcorn type and kernel count whenever the stove type changes.', 'Use one stove only and test several popcorn varieties.', 'Use different pots and durations for the two stoves without recording those differences.'],
  'Only the target varies stove type while holding the specified confounders fixed and repeating the comparison.'),
 'logic_5': (
  'The revised task asks what material is transmitted between Tasmanian devils in devil facial-tumour disease, rather than asking for a broad disease category. Studies identify the tumours as a transmissible cancer cell lineage passed during biting. What material is transferred to the recipient?',
  ['Living tumour cells from another devil.', 'Only a free-living bacterial population.', 'Only an infectious virus with no transferred tumour cells.', 'An inherited germline mutation transmitted only from parent to offspring.'],
  'The transmitted allograft consists of viable donor cancer cells, rather than an exclusively bacterial, viral, or germline mechanism.'),
 'logic_37': (
  'A supplied solar-system diameter table lists Mercury 4,879 km, Venus 12,104 km, Earth 12,742 km, Mars 6,779 km, Jupiter 139,820 km, and Saturn 116,460 km. Compare Jupiter and Saturn (the gas giants) with the four listed rocky planets. Which size relationship follows from the table?',
  ['Each of the two listed gas giants is larger in diameter than every listed rocky planet.', 'Each listed rocky planet is larger than both listed gas giants.', 'All six listed planets have the same diameter.', 'Saturn is smaller in diameter than Mercury.'],
  'The smaller listed gas giant (116460 km) exceeds the largest listed rocky planet (12742 km). The revised question concerns size rather than distance from the Sun.'),
 'logic_140': (
  'In the revised task, an early windshield wiper is observed moving a blade across the glass to remove rain. What physical action does the wiper perform, rather than what broader safety benefit might follow?',
  ['It moves a blade across the windshield to clear precipitation.', 'It changes the chemical composition of the rain.', 'It prevents clouds from forming over the car.', 'It changes the speed of the vehicle engine.'],
  'The target names the observed blade movement and removal; the alternatives do not describe that mechanism.'),
 'logic_181': (
  'A controlled decomposition trial lowers temperature from 15 degrees C to 5 degrees C. Decomposer abundance, predator abundance, and biodiversity are held constant. Which listed factor changes in this trial?',
  ['Temperature decreases.', 'Decomposer abundance increases.', 'Predator abundance decreases.', 'Biodiversity increases.'],
  '5 < 15, and the other quantities are held constant. The revised question identifies the manipulated variable without asserting a general causal conclusion from a single trial.'),
 'logic_104': (
  'The revised laboratory task is to protect eyes against particles arriving from both the front and sides. Which listed item provides enclosing eye protection rather than respiratory protection or darkening the view?',
  ['Sealed safety goggles enclosing the eyes.', 'A breathing mask covering only the nose and mouth.', 'Ordinary sunglasses with open sides.', 'A hot plate on the laboratory bench.'],
  'Only enclosing safety goggles address the specified eye exposure; the earlier breathing-mask answer addresses a different body part.'),
 'logic_60': (
  'The reactant in the revised task is sodium bicarbonate (NaHCO3), not sodium hydroxide. It reacts completely with hydrochloric acid. Which list includes all products in NaHCO3 + HCl -> products?',
  ['NaCl + H2O + CO2', 'NaCl + H2O only', 'NaCl + H2 only', 'NaOH + CO2 only'],
  'NaHCO3 + HCl -> NaCl + H2O + CO2 balances Na=1, C=1, H=2, Cl=1, O=3. Omitting carbon dioxide fails carbon conservation.'),
 'logic_18': (
  'After logging, a survey finds 120 replacement trees in a tract where no seedlings were planted and 500 in an otherwise comparable seeded tract. The revised question asks only for the observed replacement-tree comparison, not future board prices. Which statement is supported?',
  ['Fewer replacement trees are growing in the unseeded tract.', 'More replacement trees are growing in the unseeded tract.', 'Both tracts have equal replacement-tree counts.', 'The survey directly determines next year\'s board prices.'],
  '120 < 500. Tree counts do not directly determine prices.'),
 'logic_135': (
  'The revised question concerns the present-day relationship of the east coast of South America and the west coast of Africa. A supplied present-day map shows the Atlantic Ocean between them. Which description matches that map?',
  ['The two coasts are separated by the Atlantic Ocean.', 'The two coasts are joined as a single continuous landmass.', 'Both coasts lie at the North Pole.', 'No ocean lies between the two coasts.'],
  'The supplied present-day map establishes Atlantic separation; past fossil evidence addresses an earlier time.'),
 'logic_78': (
  'After a push ends, an object moves in a straight line at constant velocity across a horizontal frictionless surface. Its weight is balanced by the surface normal force and no horizontal force acts. For this later interval, which statement about the net force is correct?',
  ['The net force is zero; the forces are in equilibrium.', 'A nonzero net horizontal force is accelerating the object.', 'An unbalanced downward force is accelerating the object.', 'The net force must point forward because the object is moving.'],
  'Constant velocity means zero acceleration, and the stated vertical forces balance. F_net = m*a = 0 during the later interval.'),
 'logic_278': (
  'The school in the revised scenario is in New Zealand in the Southern Hemisphere. The task asks about the orientation of Earth during that hemisphere\'s summer, rather than summer in the Northern Hemisphere. Which orientation produces Southern Hemisphere summer?',
  ['The Southern Hemisphere is tilted toward the Sun.', 'The Northern Hemisphere is tilted toward the Sun.', 'Neither hemisphere receives sunlight.', 'Earth\'s axis loses all tilt.'],
  'Summer in a given hemisphere corresponds to that hemisphere being tilted toward the Sun; the revised location changes the hemisphere being asked about.'),
 'logic_4': (
  'On the Moon, a 1.0 kg object and a 5.0 kg object each descend by the same vertical distance of 1.0 m in the same gravitational field. The revised task asks about the gravitational potential energy lost, not the fraction of final kinetic energy acquired. How do the losses compare?',
  ['The 5.0 kg object loses five times as much gravitational potential energy as the 1.0 kg object.', 'Both lose the same gravitational potential energy.', 'The 1.0 kg object loses five times as much gravitational potential energy.', 'Neither loses gravitational potential energy.'],
  'Delta U = m*g*h. The common g and h cancel in the ratio, leaving 5/1.'),
 'logic_145': (
  'A regional simulation runs for two billion years. It specifies continuous erosion of initially high mountains, no formation of new mountains, and a final landscape with 95% of the region at sea level. Which description matches this simulated outcome?',
  ['The mountains wear down until most of the region is at sea level.', 'New high mountains continuously replace the eroded mountains.', 'Every mountain grows higher throughout the simulation.', 'The final region contains no land near sea level.'],
  'The simulation explicitly excludes new mountain formation and specifies 95% at sea level. This is a hypothetical regional scenario, not a claim about global tectonics.'),
 'logic_210': (
  'A log burns inside a sealed container that retains all solids, liquids, and gases. No matter or energy enters or leaves the entire container. In the ordinary chemical mass-conservation model, what happens to the mass of the entire closed system, rather than the mass of the remaining log alone?',
  ['The total mass of the closed system remains the same.', 'The total mass drops by the mass of all gases formed.', 'The total mass increases because heat is produced.', 'The total mass becomes zero when the log burns.'],
  'All reaction products remain inside the closed system. Chemical rearrangement changes composition, not the total conserved mass in the specified model.'),
 'logic_72': (
  'The revised task uses a hypothetical measurement: sea-surface temperature rises from 26 degrees C to 29 degrees C while current speed, ocean depth, and dissolved oxygen remain unchanged. Which measured ocean condition increased?',
  ['Sea-surface temperature.', 'Ocean-current speed.', 'Ocean depth.', 'Dissolved oxygen concentration.'],
  '29 > 26 and the other conditions are unchanged. This does not claim that hurricane frequency must increase.'),
 'logic_101': (
  'A plastic rod and fur are initially electrically neutral and isolated from other charge sources. Rubbing transfers electrons from the fur to the rod, making the rod negative. The revised question asks for the final net charge of the fur, rather than how the rod acquired electrons. What is the fur\'s final net charge?',
  ['Positive, because the fur lost electrons.', 'Negative, because the fur gained electrons.', 'Neutral, despite the stated loss of electrons.', 'Its net charge is unrelated to the electron transfer in this isolated pair.'],
  'Loss of negatively charged electrons leaves the initially neutral fur positive. This target is a charge state, not a paraphrase of the original rod-transfer answer.'),
 'logic_175': (
  'At standard atmospheric pressure, bromine is a stable liquid at 25 degrees C, away from a phase-transition boundary. The revised question asks only about bromine\'s boiling point relative to 25 degrees C. Which relation is consistent with that liquid state?',
  ['Bromine\'s boiling point is above 25 degrees C.', 'Bromine\'s boiling point is below 25 degrees C.', 'Bromine\'s boiling point is exactly 25 degrees C.', 'Bromine has no boiling point at standard atmospheric pressure.'],
  'A stable liquid away from the transition is below its boiling temperature at the specified pressure. The original oxygen answer does not answer this bromine-only question.'),
 'logic_86': (
  'A ship emits sonar pulses downward and records echo travel times from the seafloor. With the speed of sound in water known, the revised task asks what can be obtained directly from these measurements, rather than what mechanism moves continents. What is the direct mapping use?',
  ['Measure water depth and map seafloor features such as ridges.', 'Directly determine the age of every seafloor rock.', 'Directly measure the chemical composition of every rock.', 'Directly establish the full mechanism of continental motion from echo times alone.'],
  'Depth is sound speed times round-trip travel time divided by two. Echo times provide bathymetry, not chemical composition, rock ages, or a complete tectonic mechanism.'),
 'logic_0': (
  'In the revised hypothetical scenario, the time for one planetary rotation relative to distant stars changes from 24 hours to 30 hours. Define a day here as exactly one such rotation. Orbital period, mass, and volume stay unchanged. How does the defined day change?',
  ['The day becomes longer.', 'The day becomes shorter.', 'The day stays the same length.', 'Only the year changes length.'],
  '30 hours exceeds 24 hours. Explicitly defining the rotational day avoids conflating sidereal and solar days.'),
 'logic_137': (
  'A study records how long observers look at red and yellow tulips under identical viewing conditions. It does not measure genes or ask for beauty ratings. Which question is directly testable with the recorded durations?',
  ['Do observers look at red tulips longer than yellow tulips under these conditions?', 'Which genes determine tulip petal color?', 'Which tulip color is objectively the most beautiful?', 'What fertilizer changes the number of chromosomes in a tulip?'],
  'Viewing durations support a comparison of viewing time; genes, normative beauty, and chromosome changes are not measured.'),
 'logic_50': (
  'The revised task starts with 20.0 g of pure hydrogen peroxide (H2O2), which decomposes completely as 2H2O2 -> 2H2O + O2. Use atomic masses H=1 and O=16. The task asks only for the mass of water produced, excluding oxygen gas. Approximately how much water is produced?',
  ['10.6 g of water; the rest of the initial mass is in oxygen gas.', '20.0 g of water; oxygen gas has no mass.', '0 g of water; only oxygen gas is produced.', '40.0 g of water; decomposition doubles the initial mass.'],
  'H2O2 has molar mass 34 and H2O has 18. The 1:1 mole ratio gives 20*18/34 = 10.588... g; mass conservation concerns water plus oxygen, not water alone.'),
 'logic_262': (
  'The revised task classifies one child\'s observed hair as curly or straight, rather than predicting inheritance from parental traits. Which available observation directly determines this classification?',
  ['The shape of the child\'s own hair strands.', 'Only the hair type of the child\'s parents.', 'The color of the child\'s shoes.', 'The child\'s height.'],
  'The child\'s strand shape is the measured phenotype being classified; parental information is not a substitute for the specified observation.'),
 'logic_36': (
  'The revised comparison uses a star whose radius is exactly twice the Sun\'s radius. Compare diameters rather than luminosity or the size of the largest stars generally. How wide is the Sun relative to this star?',
  ['The Sun has half this star\'s diameter.', 'The Sun has twice this star\'s diameter.', 'Both have the same diameter.', 'The Sun has one hundredth this star\'s diameter.'],
  'Diameter equals twice radius for both objects; the diameter ratio is therefore 1/2.'),
}

SOURCES = {
 'logic_5': ['https://www.nature.com/articles/439549a'],
 'logic_21': ['https://www.energy.gov/cmei/fuels/hydrogen-production-electrolysis', 'https://www1.eere.energy.gov/hydrogenandfuelcells/pdfs/hydrogenenergyfuture_web.pdf'],
 'logic_123': ['https://www.noaa.gov/education/resource-collections/weather-atmosphere/hurricanes', 'https://www.gfdl.noaa.gov/blog_held/42-aqua-planet-hurricanes-and-the-itcz/'],
 'logic_26': ['https://www.weather.gov/climateservices/CvW', 'https://www.weather.gov/dtx/NewClimateNormals'],
 'logic_37': ['https://science.nasa.gov/solar-system/planets/'],
 'logic_152': ['https://prod-01-alb-toolkit-climate.woc.noaa.gov/coastal-erosion'],
}

CODE = {
 'code_22': {
  'entry_point': 'filter_integers',
  'contract': 'Keep the signature filter_integers(values). The revised function keeps only EVEN values whose exact Python type is int. Exclude bool values, floats, strings, and all other types. Preserve the order and duplicates of qualifying values. Zero and negative even integers qualify. The input is a finite list of ordinary Python values.',
  'reference': 'def filter_integers(values):\n    return [x for x in values if type(x) is int and x % 2 == 0]\n',
  'rationale': 'Filtering by exact integer type alone is already implemented by the current original reference. The new parity condition changes its behavior on odd integers.',
  'witness': [[1, 2]],
 },
 'code_39': {
  'entry_point': 'prime_fib',
  'contract': 'Keep the signature prime_fib(n). The revised function returns a LIST of the first n prime Fibonacci numbers in increasing order, instead of returning only the nth number. Fibonacci numbers follow F0=0, F1=1, Fk=F(k-1)+F(k-2). A prime is an integer greater than 1 with no positive divisors other than 1 and itself. The revised input domain is an integer n with 0 <= n <= 8. For n=0 return an empty list. Each qualifying value appears once.',
  'reference': 'def prime_fib(n):\n    result = []\n    a, b = 0, 1\n    while len(result) < n:\n        a, b = b, a + b\n        if a < 2:\n            continue\n        d = 2\n        while d * d <= a and a % d != 0:\n            d += 1\n        if d * d > a:\n            result.append(a)\n    return result\n',
  'rationale': 'The returned object changes from one integer to the ordered prefix list. Independent sequence enumeration supports all nine inputs in the explicitly bounded revised domain. Original-reference quarantine is preserved.',
  'witness': [1],
 },
 'code_44': {
  'entry_point': 'change_base',
  'contract': 'Keep the signature change_base(x, base). The revised input domain is nonnegative integer x and integer base with 2 <= base <= 9. Return a string of base-converted digits in LEAST-SIGNIFICANT-FIRST order instead of the usual most-significant-first order. Preserve all digits, including zeros at the start of the reversed string. For x=0 return "0". Negative x and bases outside 2..9 are outside this revised task.',
  'reference': 'def change_base(x, base):\n    if x == 0:\n        return "0"\n    digits = []\n    while x:\n        x, digit = divmod(x, base)\n        digits.append(str(digit))\n    return "".join(digits)\n',
  'rationale': 'Reversing digit order changes positive-input behavior; the explicit nonnegative domain and zero rule remove the old undefined negative-input requirement.',
  'witness': [6, 2],
 },
 'code_49': {
  'entry_point': 'modp',
  'contract': 'Keep the signature modp(n, p). Change the exponentiation BASE from 2 to 3: return 3**n modulo p. The revised input domain is nonnegative integer n and positive integer p. Return an integer in 0..p-1, including when n=0 or p=1. The old base-2 examples are superseded.',
  'reference': 'def modp(n, p):\n    return pow(3, n, p)\n',
  'rationale': 'The current original reference already reduces the n=0 result modulo p. Changing the exponentiation base to 3 instead requires a real functional update.',
  'witness': [1, 5],
 },
 'code_76': {
  'entry_point': 'is_simple_power',
  'contract': 'Keep the signature is_simple_power(x, n). The revised input domain is nonnegative integers x and n. Return a Python bool that is True exactly when x == n**k for some STRICTLY POSITIVE EVEN integer exponent k (2, 4, 6, ...). Exponent zero and odd exponents do not qualify. For n=0, such powers equal 0; for n=1, such powers equal 1. Supersede conflicting earlier examples.',
  'reference': 'def is_simple_power(x, n):\n    if n == 0:\n        return x == 0\n    if n == 1:\n        return x == 1\n    step = n * n\n    power = step\n    while power < x:\n        power *= step\n    return power == x\n',
  'rationale': 'Requiring positive even exponents changes behavior for odd powers and for exponent zero, while fully specifying bases zero and one.',
  'witness': [8, 2],
 },
}
