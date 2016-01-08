from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from nltk.classify import NaiveBayesClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import math, time
import logging
import os
import pickle
import re


logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s][%(name)s][%(asctime)s] %(message)s')


class Classifier:
    def __init__(self, saveResult=True, model_name="model"):
        self.preprocessing = Preprocessing()
        self.logger = logging.getLogger("classifier")
        self.logger.setLevel(logging.DEBUG)
#        file_handler = logging.FileHandler(filename=os.path.join(LIB_PATH, "../log/log.txt"))
#        file_handler.setFormatter(logging.Formatter('[%(levelname)s][%(name)s][%(asctime)s] %(message)s'))
#        self.logger.addHandler(file_handler)
        self.tags = self.api.getSignalsTags()
        t1 = time.time()
        self.logger.info("Building model on start...")
        self.vectorizer = self.build_vectorizer()
        self.model = self.learn(saveResult, model_name)
        t2 = time.time()
        self.logger.info("Model was built in: %ss" % str(t2-t1))

    def build_vectorizer(self):
        vectorizer = CountVectorizer(min_df=4,
                                     binary=True,
                                     token_pattern=token_pattern,
                                     stop_words=STOPWORDS)
        return vectorizer

    def content(self, signal, serialize=False):
        ini_content = ((signal.get("title", "") or "") + " " + (signal.get("content", "") or ""))
        self.logger.debug("Trying to retrieve serialized signal %s..." % signal.get("id",""))
        fname = os.path.join(RESOURCES_PATH, "lemmatized_signals/signal_%s.pickle" % signal.get("id","").replace(":", ""))
        if os.path.exists(fname):
            self.logger.debug("A corresponding file was found, loading it...")
            try:
                with open(fname, "rb") as f:
                    return pickle.load(f)
            except:
                self.logger.debug("Could not load serialized signal, computing it.")
        self.logger.debug("Lemmatizing input signal: \"%s\"" % ini_content)
        preprocessing = self.preprocesser.preprocess(ini_content)
        self.logger.debug("Finished lemmatizing input signal: %s" % str(preprocessing))
        if serialize:
            with open(fname, "wb") as f:
                pickle.dump(preprocessing, f)
        return preprocessing

    def get_learn_docs(self):
        validated = self.api.getValidated() # List<Signal>
        docs = {}
        for doc in validated:
            docs[doc["id"]] = doc
            
        return docs

    def learn(self, saveResult=True, model_name='model'):
        self.logger.info("Try to load a model if one exists...")
        fname = os.path.join(RESOURCES_PATH, "classification_model/%s.pickle" % model_name)
        try:
            with open(fname, "rb") as f:
                model = pickle.load(f)
            with open(os.path.join(os.path.dirname(fname), "vectorizer.pickle"), "rb") as f:
                self.vectorizer = pickle.load(f)
            return model
        except IOError as e:
            self.logger.warning("I/O error: {0}".format(e))
            pass
        except EOFError as e:
            self.logger.warning("EOF error: %s" % e)
            pass
        else:
            self.logger.info("Failed to load an existing model.")
            pass

        docs = self.get_learn_docs()
        self.logger.info("Training set fetched (%d documents)." % len(docs))

        if len(docs) < 100:
            self.logger.warning("Not enough documents to learn anything.")
            return None
        keys = list(docs.keys())
        docs = [docs[key] for key in keys] # transform the dictionary docs into List<Signal>

        def validation(doc):
            validated = doc["validatedTags"] or []
            return tuple([tag in validated for tag in self.tags])

        validation = [validation(doc) for doc in docs]
        validation = np.array(validation) # Transform List to Array

        #content = [self.content(doc) for doc in docs]
        content = []
        chrono_list = []
        for doc in docs:
            preprocessing = self.content(doc, serialize=True)
            content.append(preprocessing[0])
            chrono_list.append(preprocessing[1])

        self.logger.info("Lemmatization duration per signal: %s" % chrono_list)
        self.logger.info("Lemmatization mean duration: %s" % np.mean(chrono_list))
        self.logger.info("Lemmatization quartiles: %s" % str([np.percentile(chrono_list, k) for k in [0, 25, 50, 75, 100]]))

        self.logger.info("Vectorizing lemmatized documents...")
        docs = self.vectorizer.fit_transform(content).toarray()
        self.logger.info("Done vectorizing lemmatized documents.")

        nfolds = 10
        self.logger.info("Training classifier with %d exammples and %d folds..." % (len(docs), nfolds))

        for idx, tag in enumerate(self.tags):
            tag_validation = [v[idx] for v in validation]
            sFolds = StratifiedKFold(tag_validation,
                                     n_folds=nfolds)
            labels_cm = []
            predictions = []

            for train_index, test_index in sFolds:
                X_train = docs[train_index]
                y_train = [tag_validation[i] for i in train_index]
                X_test = docs[test_index]
                labels_cm += [tag_validation[i] for i in test_index]
                model = LogisticRegression()
                #model = ExtraTreesClassifier()
                #model = MultinomialNB()
                #model = GaussianNB()
                #model.fit(X_train, y_train)
                model.fit(X_train, y_train)
                #model.transform(X_train)
                lpredictions = model.predict(X_test)
                predictions += list(lpredictions)

            report = classification_report(labels_cm, predictions,
                                           labels=[True, False])
            self.logger.info("Cross-validated performance of tag [%s]:" % tag)
            self.logger.info(report)

        model = OneVsRestClassifier(LogisticRegression())
        #model = OneVsRestClassifier(ExtraTreesClassifier())
        #model = OneVsRestClassifier(MultinomialNB())
        #model = OneVsOneClassifier(GaussianNB())
        model.fit(docs, validation)

        feature_names = self.vectorizer.get_feature_names()

        count = 0

        for idx, estimator in enumerate(model.estimators_):
            self.logger.info("TAG %s" % (self.tags[idx]))
            if hasattr(estimator, "coef_"):
                for coefs in estimator.coef_:
                    coefs = list(sorted(zip(coefs, feature_names),
                                           reverse=True))
                    coefs = [(e[1], "%.2f" % e[0]) for e in coefs]
                    self.logger.info(coefs[:20])
                    count += 1
                    self.logger.info("")
            self.logger.info("============")
            self.logger.info("")

        prediction = model.predict(docs)

        for tag_idx in range(len(self.tags)):
            self.logger.info(self.tags[tag_idx])
            confusion = confusion_matrix(validation[:, tag_idx],
                                         prediction[:, tag_idx])
            self.logger.info(confusion)
            if confusion.shape != (2, 2):
                continue

            if saveResult:
                self.api.saveConfusion({
                    "className": self.tags[tag_idx],
                    "truePositive": int(confusion[0][0]),
                    "falsePositive": int(confusion[0][1]),
                    "falseNegative": int(confusion[1][0]),
                    "trueNegative": int(confusion[1][1]),
                })
        self.logger.info("\n[[TP FP]\n [FN TN]]")

        with open(fname, "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(os.path.dirname(fname), "vectorizer.pickle"), "wb") as f:
            pickle.dump(self.vectorizer, f)

        return model

    def tag(self, signal):
        blacklist = 'blacklist'
        self.logger.info("Try to load the blacklist...")
        fname = os.path.join(RESOURCES_PATH, "blacklist/%s.pickle" % blacklist)
        try:
            with open(fname, "rb") as f:
                blacklist = pickle.load(f)
        except IOError as e:
            self.logger.warning("I/O error: {0}".format(e))
            self.logger.info("Failed to load the blacklist.")
            pass
        except EOFError as e:
            self.logger.warning("EOF error: %s" % e)
            self.logger.info("Failed to load the blacklist.")
            pass

        if blacklist is not None:
            if "sourceId" in signal and signal["sourceId"] in blacklist:
                return {"id": signal["id"], "tags": []}

        if "sourceId" in signal and re.search("TWITTER", signal["sourceId"]) and re.match("RT", signal.get("content","")):
            return {"id": signal["id"], "tags": []}

        if self.model is None:
            self.logger.info("No model")
            return {"id": signal["id"], "tags": []}

        #content = self.vectorizer.transform([self.content(signal)[0]])

        #if content[0].getnnz() < 5:
        #    self.logger.info("Not enough tokens")
        #    return {"id": signal["id"], "tags": []}

        #prediction = self.model.predict(content)[0]
        #tags = [tag for pred, tag in zip(prediction, self.tags)
        #        if pred]
        # Needed for the "IndexedSignal" model in the jbm
        #return {"id": signal["id"], "tags": tags}


        content = self.vectorizer.transform([self.content(signal)[0]])
        prediction = ["%.2f" % e for e in self.model.predict_proba(content)[0]]
        maxi = zip([self.tags[0]], [prediction[0]])
        for tag, pred in zip(self.tags, prediction):
            tag_max, pred_max = zip(*maxi)
            tag_max = list(tag_max)
            pred_max = list(pred_max)
            if float(pred) > float(pred_max[0]):
                maxi = zip([tag], [pred])
            else:
                maxi = zip([tag_max[0]], [pred_max[0]])
        maxi = list(maxi)[0]
        if float(maxi[1]) > 0.5:
            return {"id": signal["id"], "tags": maxi[0]}
        else:
            return {"id": signal["id"], "tags": []}

    def explain(self, text):
        from sklearn.multiclass import _ConstantPredictor
        _ConstantPredictor.predict_proba = \
            lambda s, X: np.tile(np.repeat(s.y_, X.shape[0]), (1, 2))

        if self.model is None:
            return "No model"
        #content = self.vectorizer.transform([RegexpCleaner().clean(text)])
        content = self.vectorizer.transform([self.preprocesser.preprocess(text)[0]])
        prediction = ["%.2f" % e for e in self.model.predict_proba(content)[0]]

        self.logger.info(list(zip(self.tags, prediction)))

        maxi = zip([self.tags[0]], [prediction[0]])
        for tag, pred in zip(self.tags, prediction):
            tag_max, pred_max = zip(*maxi)
            tag_max = list(tag_max)
            pred_max = list(pred_max)
            if float(pred) > float(pred_max[0]):
                maxi = zip([tag], [pred])
            else:
                maxi = zip([tag_max[0]], [pred_max[0]])

        #tag, prediction = zip(*maxi)
        #prediction = self.model.predict(content)[0]
        self.logger.info(list(maxi)[0])

