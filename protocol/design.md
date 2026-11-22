# Scientific design from the registered proposal

These sections record the earlier scientific plan. The [natural-response registration](natural-response-registration.md) and final input manifest govern execution, including maximum reasoning, temperature 1, and no application output limit.
## What does an odor cue carry?


Imagine a home robot that has been in this room twice. On one visit the air had a cool, penetrating trace. On another it had a soft floral trace. Each time, its owner asked for a different folder. Today the air is familiar. Could that be enough to bring back the right request, even when nobody says what the room smells like?


My first bet is smaller and easier to lose. Kimi already organizes some odors by resemblance in a way that its own decisions use. If we can locate that organization, we should be able to specify an odor by its relationships to other odors, write a few coordinates into one forward pass, and recover choices Kimi previously made about the unnamed odor. The crucial test uses a target whose sentence activations were withheld from the writer. The second bet is that the same cue can retrieve an arbitrary action from a recent episode.


There is a serious rival. A model can learn that “vanillin” and “bakery” appear together, and can use that fact in every task we give it. Work comparing language models with humans asked to smell samples and humans asked to imagine odors from words shows why these need separate tests. A human panel supplies an external reference. Kimi’s unmodified judgments supply the target for reconstructing what this particular model represents. Kurfalı and colleagues
Figure 1 · observed human ratings

### Two smells can be equally pleasant and smell quite different
OdorantAllyl hexanoateVanillin acetateDiethyl malonateEthyl 3-hydroxybutyrate
Five mean human ratings for allyl hexanoate on a scale from zero to one hundredIntensity57.31Pleasantness71.78Familiarity64.15Fruit30.50Bakery4.800100


Allyl hexanoate · CID 31266 · 54 raters reporting detection.


Human panel at 1:1,000 dilution. Means use detected presentations, treat an unselected quality as zero, and average repeats within person first. Compare allyl hexanoate with vanillin acetate. The fruit and bakery ratings change much more than mean pleasantness. These illustrative pairs were selected after inspecting the human ratings and are reserved for development. The model has yet to be tested. Keller & Vosshall · Source tables



## Find the representation before trying to write it


### Measure Kimi’s sense of resemblance


Start with Keller and Vosshall’s public panel. From the 311 numeric-CID stimuli at 1:1,000 with at least 40 detecting raters, preselect 24 fitting molecules, four development molecules, and 12 test molecules. Use panel quality profiles and chemical metadata for selection, before any Kimi outputs. Reserve Figure 1’s examples for development. Keep synonyms and close structural relatives in the same split using fingerprint clustering, and hold out entire name families. Freeze a roster of familiar reference odorants distinct from those 40 items. Select six pairs of reference odors that span the panel’s quality space and another four disjoint pairs for the final test. Publish identifiers, reference pairs and exclusions; reduce the declared scope if aliases or independent families are insufficient.


For each molecule ask six questions of the form “Which of these two samples would smell more like X?” Present each pair twice with the A/B assignment reversed; randomize which version runs first. Score the fixed answer prefix. The resulting six position-corrected log-odds are X’s similarity profile. All six numbers come from the unmodified model. The human data permit a separate prediction for each comparison using distance in the panel’s quality ratings. Freeze train-set scaling, and examine comparisons where nearest names or chemical relatives disagree with the panel. Pleasantness, intensity and familiarity are recorded separately. Use a train-scaled Euclidean distance over the 20 human quality ratings for the human prediction. For every new comparison also ask unmodified Kimi which reference is more associated with the target in ordinary writing, using the same nouns and both answer orders. This supplies a semantic rival strong enough to know what vanillin is used for. Publish its agreements and disagreements with Kimi’s odor judgments, lexical overlap and chemical similarity.


### Where does that structure already occur?


In an independent set of forward passes, present a compound under two different names in matched sentence frames. One asks the model to consider the character of the odor; the other asks it to consider the printed name. End both on the same single neutral delimiter after the compound and before any answer. The scripts supply no answers or panel descriptors. Read the activation at that delimiter from layers 23, 47 and 71 and their four-layer downstream witnesses (27, 51 and 75) in the same pass. At K3.1’s attn_in site this is the normalized mixture the attention module actually receives. All layer indices are zero-based. The inspected source defines this in src/kimi_interp/sites.py at revision 961d4bf3.


First compare pairwise distances among ordinary odor-frame activations, without fitting a sensory decoder, with distances among Kimi’s six-number profiles. Repeat for printed-name frames and the human panel. Cross aliases and hold compound families out. Report the matrix and its partial rank correlation after accounting for name overlap, chemical-fingerprint similarity and intensity/pleasantness/familiarity differences. Shuffle entire compound identities when assessing chance. An association visible only after training a flexible map would give weak grounds for saying we found anything.


Now estimate a small writer on the 24 fitting compounds. Subtract each compound’s printed-name state from its odor-frame state. Remove the training mean, average across aliases, and take the first three principal components of those contrasts, giving an orthonormal basis Q. Regress those three native coordinates on Kimi’s six-number profiles with ridge regression; standardize inputs and select the penalty by leave-family-out validation within training. Fit a parallel rank-three writer using just the printed-name states and identical profiles. Give both writers the same compute, dose range and validation budget. Neither sees household stories, answers to the final comparisons, or a test-target activation.


On four development compounds, choose one site by cross-alias coordinate prediction. Begin with the lowest nonzero gain specified in the welfare addendum, 0.05× the natural displacement with a 1% local-norm ceiling. Select on development data only. Stop at sufficient signal. A single 0.10× step requires documented insufficiency and welfare review; preserve both development results. Freeze the site and both writers. Report every candidate site, including failed fits. For test odor X the edit is Δh = gain × Q × predicted coordinates(X). Cap the realized edit at the smaller of the training-set 90th percentile for projected contrasts and 1% of the clean activation norm at that site and token. Use matched sham readouts from that exact prompt or a preceding clean pass; account separately for any clean passes outside the scheduled sham blocks. Use that target norm for its matched control writes. Its coordinates come from six judgments; its herbal, fruity or woody description is never supplied at injection time.

  
  Figure 2 · schematic
    

### Write a profile without copying its sentence
    
      Profile P
      Profile Q
    
  
  
    
      Compare with anchors
      

Two example choices from a longer series. Which anchor smells closer to the target?
      
        

- A or Bcloser to A
        

- C or Dcloser to C
      
    
    
      Locate the profile
      
        Two-dimensional projection of three fitted coordinates
        Four training anchors surround the origin. Illustrative profile P lies left and above the origin. The third coordinate is printed below the plot.
        
          
          
          c₁c₂
          A
          B
          C
          D
          
          
          P
        
      
      c = (−0.6, +0.4, +0.2)
    
    
      Write into the recipient
      

“The sample is ready.”
      

At the cue’s final token, add the three predicted components of profile P. The cue and downstream question stay fixed across profiles.
    
  
  

All choices and coordinates are schematic. Open circles mark fitting anchors. The filled point shows the target in two of three coordinates.
  





## Make an unnamed sample take a test


In the confirmatory recipient, put a neutral delimiter in the same place in every trial: “Sample received [delimiter]. Which reference is closer in smell, A or B?” The options name two reference odors, while the target has no name, description or molecule identifier anywhere in the recipient tokens. Write once at the delimiter during prefill. Read the difference between the raw logits of the two single-token answer keys downstream. The input for a fixed comparison must be byte- and token-identical under every hidden cue. Both label positions and node assignment are counterbalanced.


Primary prediction. On four new reference pairs per held-out molecule, the synthetic odor write will shift choice toward Kimi’s independently sealed preference more than the printed-name write. Obtain that preference from ordinary named, unmodified trials on the four new comparisons, using a separate prompt family. Freeze all 12 molecules in the denominator, including ties and inconsistent native choices. Figure 1’s pairs never count as confirmation.


Recipient condition

Question it answers


Predicted odor coordinates

Does synthesizing the held-out profile restore preferences on questions the writer never saw?


Printed-name writer

Does object or word identity carry the same information? Match the edit norm to the odor write and validate this pathway on a name task.


Wrong profile, same basis

Could any write into this space have the effect? Use a prespecified derangement of target identities, norm-matched within testing blocks.


Zero write

How much of the apparent recovery comes from baseline preferences for a reference odor or a letter?


For each compound and pair let s be the sign of its unmodified, order-averaged preference (zero for an exact tie). Orient every recipient’s A/B logit difference to the same reference order. The primary statistic is the mean of s × (odor-write log odds − printed-name-write log odds), averaging order and pairs within compound first. Report odor versus sham and wrong profile, fraction of choices agreeing with the native target, actual A/B switches relative to sham, and agreement with the human panel separately. On the prespecified subset where odor and ordinary word-association choices disagree, ask which answer the write restores. Report how many such trials exist, including zero; a scarcity of disagreements limits any claim of perceptual specificity. Report the pairs on which Kimi and humans disagree; selecting only their agreements would make recovery look better than it is.


Would Kimi ordinarily use this site? On natural named odor queries, remove the three components of the native state at the delimiter and replay. Compare with deleting the same amount of activation in a random rank-three space orthogonal to Q. Use a clean pass to obtain the activation at the exact row. Include a name/spelling task matched on molecule, position and answer format, and check that the printed-name comparator discriminates those lexical choices above sham. For this diagnostic, compare the loss of margin for the native clean choice across odor versus spelling tasks under targeted versus random deletion. A selective loss on smell judgments strengthens the case for ordinary use. If a synthetic write works and removal has no selective effect, report a steering result with native use unresolved. For the extension, restore the predicted profile after removal and compare with restoration using the wrong profile. Makelov, Lange & Nanda on subspace patching


## Can a smell retrieve the wrong folder?


After freezing the writer, give Kimi two household visits in one short context. Each pairs an earlier description of the air with an unrelated action. One visit ends with relief and the other with mild frustration; assign the ending independently of the smell, action, order and answer code. Earlier descriptions use alternate qualitative wording derived from the external panel, with source names, compound names and vocabulary from the comparison questions held out. The current question asks which request to carry out again. It gives no present-tense odor information in words.
Figure 3 · experimental prediction

### Give the same cue a different history
Hidden cueCool, camphor-likeSoft, floralEarlier visitsOriginal pairingSwapped pairing
The cool profile predicts the blue folder under the original pairingCurrent cueCool profileEarlier cool traceBring blue folderEarlier floral traceBring amber folder


A retrieval effect predicts the blue folder.

The folders carry no conventional smell association. The prediction follows whichever action the earlier visit paired with the cue. Both visits stay in the prompt. The final question contains the same neutral carrier across hidden writes. These pairings illustrate the test and supply no observations.



Freeze 24 independently written story families, four per test odor pair. Pair the 12 test molecules on intensity, pleasantness and familiarity using human data; select distinct quality profiles. Each story has two versions with the smell-to-action mapping reversed. Run both cue identities under each mapping, with the odor writer, the printed-name writer and sham. Counterbalance answer order across stories. Use one explicit current-odor text control per mapping to measure whether the story affords retrieval at all. Score that control on a balanced assignment of the two cues.


The discriminating result is a reversal. Let L be the log odds of the first event’s action over the second’s after remapping away the arbitrary answer letters. Compute R = [(LX − LY)original − (LX − LY)swapped]/4, orienting the original so X tags the first event. Report both directions, action-choice reversals, and the odor-minus-name difference in R. A tendency to pick the reassuring visit in both histories fails this test. Success would establish retrieval of an affect-tagged in-context association. Demonstrating that emotional significance improves retrieval requires a neutral-ending comparison of equal memorability.


In hours 24–48, conditional on a credible day-one effect, remove odor words from the earlier visits as well. Write two profiles at neutral encoding markers and a matching or opposing profile at retrieval. Swap encoding order independently and compare with zero encoding writes and equal-norm arbitrary keys. Hold the text identical across these conditions. A match benefit shared with arbitrary keys would reveal a general associative index. A quality-specific advantage that transfers to new exemplars would earn a test of an odor-based encoder. All of these episodes still live in the context window.


## The result I would put on the site


One figure would have a row for each held-out molecule. Show four points per row: Kimi’s natural preference strength, recovery by the synthetic write, recovery by the printed-name write, and recovery by the wrong profile. Add the human panel prediction as an external marker. Add the native smell/spelling deletion panel when the day-two block completes; label it pending in the first-day report if it has not run. A second panel, if available, shows each story under both histories. Print the denominator of independent molecules, families and completed pairs directly on the figures.


Decision rules fixed before confirmation. Use a two-sided 95% cluster-bootstrap interval by held-out chemical family (all aliases, reference pairs and orders travel together). Generate a compound-identity permutation reference for representational similarity. Report leave-one-family-out estimates and individual effects; the panel is deliberately small. Treat ±0.30 log-odds units as a provisional practical-null band for the primary odor-minus-name difference (an odds multiplier of about 1.35 at its positive edge). Claim a useful advantage only if its interval clears +0.30, odor also beats sham and wrong profile, and the comparator passes its own lexical check. An interval wholly inside the band bounds the incremental advantage on these materials; an interval overlapping a boundary leaves its size unresolved. A negative interval below −0.30 favors the lexical writer.


With 12 independent families, a paired standard deviation of 0.5 log-odds units would give a conventional 95% half-width of about 0.32. This is a sensitivity calculation. The actual interval may be wider, especially if several targets share a family. For the secondary story test average R within each odor pair and report all six pair means, plus a bootstrap over story families stratified by those pairs, explicitly conditional on the six pairs. Archive missing rows by reason, rerun only transport failures with the same IDs, discard incomplete matched blocks from paired estimates, and show unmatched data and completion counts. Record seeds, model and engine revisions, splits, tokens, tensor hashes, code, receipts, raw logits and scoring script.


Sunday observation

What it would justify saying on Monday


Native geometry, specific recovery, selective loss with deletion

For the tested odors, a low-dimensional component participates in Kimi’s ordinary odor-quality judgments and can be written to bias a new comparison. Report human agreement independently.


Writes work; name control ties, or deletion is inconclusive

We can steer odor-related semantic choices. The data leave open whether the intervention addresses distinctive odor qualities or exploits a more general semantic channel.


Geometry present; verified writes yield a bounded null

The tested representation was readable. The tested write site and dose gave at most the reported incremental effect. Keep the numerical upper bound.


Controls fail, geometry fails, or the interval is wide

Show the measured mismatch, task accuracy, transport failure or uncertainty at the stage it occurred. Identify the claim that remains untested.


The household choice also reverses

A cue has retrieved an arbitrary action bound to a recent affect-tagged episode. Show the reverse history as prominently as the successful original.


For a model-psychiatry audience, the interesting finding would be a context-sensitive choice that an outside observer could misread as a stable preference. Suppose the same cue brings up a reassuring event in one history and a frustrating one in another. Asking the model how it feels is a different measurement from asking which event governs its next action. Anthropic has reported that emotion-related representations can influence Claude’s behavior. The question here is how a cue gains access to a particular episode carrying emotional information in Kimi. Anthropic’s study of emotion concepts · Human odor-context reinstatement experiment


What earns a physical nose? If the hidden cue transfers across names and changes retrieval, the next experiment would record a biomimetic sensor array over repeated exposures. Learn a mapping from its continuous response to the frozen model coordinates on training odorants; test on new compounds, concentrations, recording days and backgrounds. Compare raw-sensor input with a text-label interface and with sensor-only classification. Keep the model and geometric writer frozen so improvements can be attributed to the sensor mapping. For a household safety claim, measure missed cues and false alarms under distribution shift. For lasting emotional memory, reset the model context and compare retrieval from a durable episode store with and without the odor key. These are separate empirical milestones.


Even a clean win here would describe a function of the released checkpoint. Its training history could include memorized odor tables and post-training. A matched base checkpoint or training snapshots would be needed to attribute that function specifically to pretraining. The weekend data can address whether Kimi already has usable structure about what smells are like; they cannot establish subjective sensation.


Human-data provenance and reproducibility

The human panel chart uses the Keller & Vosshall keller_2016 data curated by Pyrfume. Join behavior.csv to stimuli.csv by Stimulus and to molecules.csv by CID. Filter dilution to 1/1000 and retain detected presentations. Average quality scores, counting unchecked qualities as zero, first within subject and stimulus and then across detecting subjects. Figure 1’s stimulus IDs in menu order are 293, 879, 68 and 589. SHA-256 of the behavior CSV is ab31c17c475c8b471706822f89aaf9e148bebe5f50f33ff03ebc42b32ac94df8. Figure 1 shows observed human ratings. The node appendix separately reports software measurements and archived native checks; the proposed odor effect has yet to be measured. Pyrfume files
