import os
import csv
import shutil
import time
import vectorizer
import neural
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

"""
Evaluate the IR subsystem and save the results.
Contains different variations for the IR subsystem.
"""
preprocess = vectorizer.preprocess

# Evaluate the performance of the IR system
def eval_ir(vector, model_knn, keys, queries, document_ids, k, ng, svd, n_components):

    query_results_file = f'query_results_{k}_{keys}' + ".csv"
    
    overall_summary_file = f'overall_summary_{k}_{keys}' + ".csv"
    if ng:
        output_directory = f'./results/ngram/{k}/{keys}'
    elif svd:
        output_directory = f'./results/svd/svd_{n_components}/{k}/{keys}'
    else:
        output_directory = f'./results/{k}/{keys}'

    if os.path.exists(output_directory) and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    query_results_file = os.path.join(output_directory, query_results_file)
    with open(query_results_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Distances", "Precision", "Recall", "F1_Score"])
    accuracy_values = []
    distance_values = []
    precision_values = []
    correct_values = []
    recall_values  = []
    f1_values = []
    ap_values = []
    for query_dict in queries['queries']:
        query = query_dict['q']
        relevant_docs = set(query_dict['r'])
        if k > 0:
            query = preprocess(query)

            if ng:
                query_ngram = [' '.join(query[i:i+2]) for i in range(len(query)-1)]  # Convert to bigrams
                query = query + query_ngram #ngram + individual words
            
            query_vector = vector.transform([" ".join(query)])
            if svd:
                query_vector = svd.transform(query_vector)

            distances, indices = model_knn.kneighbors(query_vector, n_neighbors=k)

            retrieved_docs = {document_ids[i] for i in indices.flatten()}
            count = 0
            correct = 0
            for i in retrieved_docs:
                if i in relevant_docs:
                    correct = 1
                    count +=1
            
            accuracy = count / len(retrieved_docs) 

            #only evaluate queries that have a solution 
            if relevant_docs :
                distance_values.append(distances.mean())
                accuracy_values.append(accuracy)
                correct_values.append(correct)

                # Precision
                tp = len(relevant_docs & retrieved_docs)
                fp = len(retrieved_docs) - tp
                precision = tp / (tp + fp)
                precision_values.append(precision)
                # Recall
                recall = tp / len(relevant_docs)
                recall_values.append(recall)
            
                # F1 Score
                beta = 1
                f1 = 0
                if recall or precision:
                    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                f1_values.append(f1)
 
                # AP
                retrieved_docs = [document_ids[i] for i in indices.flatten()]
                ap = 0
                count = 0
                if relevant_docs and tp:
                    for i in range(len(retrieved_docs)):
                        if retrieved_docs[i] in relevant_docs:
                            count +=1
                            ap += (count/(i+1))
                    ap = ap/tp
                ap_values.append(ap)

                with open(query_results_file, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([query_dict["q"], 
                                     [[round(val, 4) for val in sublist] for sublist in distances],
                                     round(precision, 4), 
                                     round(recall, 4), 
                                     round(f1, 4)])
                    

    average_distance = sum(distance_values) / len(distance_values)
    average_precision = sum(precision_values) / len(precision_values)
    average_correct = sum(correct_values) / len(correct_values)
    average_recall = sum(recall_values) / len(recall_values)
    average_f1 = sum(f1_values) / len(f1_values)
    average_map = sum(ap_values) / len(ap_values)
    
    overall_summary_file = os.path.join(output_directory, overall_summary_file)
    with open(overall_summary_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance", "MAP", "Precision", "Recall", "F1_Score", "Correct"])
        writer.writerow([round(average_distance, 4), 
                         round(average_map, 4), 
                         round(average_precision, 4),
                         round(average_recall, 4), 
                         round(average_f1, 4), 
                         round(average_correct, 4)])
    print(f'saved results to: {output_directory}')
    
# Evaluate the performance of the IR system with word embeddings
def eval_ir_word(model_knn, k, keys, queries, vector_size, anthology_sample):

    query_results_file = f'query_results_{k}_{keys}' + ".csv"
    
    overall_summary_file = f'overall_summary_{k}_{keys}' + ".csv"
    output_directory = f'./results_2/word/{k}/{keys}'

    if os.path.exists(output_directory) and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    query_results_file = os.path.join(output_directory, query_results_file)
    with open(query_results_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Distances", "Precision", "Recall", "F1_Score"])

    accuracy_values = []
    distance_values = []
    precision_values = []
    correct_values = []
    recall_values  = []
    f1_values = []
    ap_values = []
    times = []
    for query_dict in queries['queries']:
        query = query_dict['q']
        relevant_docs = set(query_dict['r'])

        query = neural.preprocess(query)
        query_document = neural.get_embedding_w2v(query, vector_size)
    
        query_document = query_document.reshape(1, -1)
        
        start_time = time.time()
        
        distances, indices = model_knn.kneighbors(query_document, k)

        end_time = time.time()
        times.append(end_time - start_time)

        retrieved_docs =  {anthology_sample[int(index)]['acl_id'] for index in indices.flatten()}
        
        count = 0
        correct = 0
        for i in retrieved_docs:
            if i in relevant_docs:
                correct = 1
                count +=1
        
        accuracy = count / len(retrieved_docs)
        #only evaluate queries that have a solution 
        if relevant_docs :
            distance_values.append(distances.mean())
            accuracy_values.append(accuracy)
            correct_values.append(correct)

            # Precision
            tp = len(relevant_docs & retrieved_docs)
            fp = len(retrieved_docs) - tp
            precision = tp / (tp + fp)
            precision_values.append(precision)
            # Recall
            recall = tp / len(relevant_docs)
            recall_values.append(recall)
        
            # F1 Score
            beta = 1
            f1 = 0
            if recall or precision:
                f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            f1_values.append(f1)

            # AP
            retrieved_docs =  [anthology_sample[int(index)]['acl_id'] for index in indices.flatten()]

            ap = 0
            count = 0
            if relevant_docs and tp:
                for i in range(len(retrieved_docs)):
                    if retrieved_docs[i] in relevant_docs:
                        count +=1
                        ap += (count/(i+1))
                ap = ap/tp
            ap_values.append(ap)

            with open(query_results_file, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query_dict["q"], 
                                 [[round(val, 4) for val in sublist] for sublist in distances],
                                 round(precision, 4), 
                                 round(recall, 4), 
                                 round(f1, 4)])

    average_distance = sum(distance_values) / len(distance_values)
    average_precision = sum(precision_values) / len(precision_values)
    average_correct = sum(correct_values) / len(correct_values)
    average_recall = sum(recall_values) / len(recall_values)
    average_f1 = sum(f1_values) / len(f1_values)
    average_map = sum(ap_values) / len(ap_values)
    average_time = sum(times) / len(times)
    
    overall_summary_file = os.path.join(output_directory, overall_summary_file)
    with open(overall_summary_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance", "MAP", "Precision", "Recall", "F1_Score", "Correct", "Speed"])
        writer.writerow([round(average_distance, 4), 
                         round(average_map, 4), 
                         round(average_precision, 4),
                         round(average_recall, 4), 
                         round(average_f1, 4), 
                         round(average_correct, 4),
                         round(average_time, 6)])
    print(f'saved results to: {output_directory}')

# Evaluate the performance of the IR system with neural embeddings
def eval_ir_neural(model_knn, k, keys, queries, model, anthology_sample):

    query_results_file = f'query_results_{k}_{keys}' + ".csv"
    
    overall_summary_file = f'overall_summary_{k}_{keys}' + ".csv"
    output_directory = f'./results_2/{k}/{keys}'

    if os.path.exists(output_directory) and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    query_results_file = os.path.join(output_directory, query_results_file)
    with open(query_results_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Distances", "Precision", "Recall", "F1_Score"])

    accuracy_values = []
    distance_values = []
    precision_values = []
    correct_values = []
    recall_values  = []
    f1_values = []
    ap_values = []
    times = []
    for query_dict in queries['queries']:
        query = query_dict['q']
        relevant_docs = set(query_dict['r'])

        query = neural.preprocess(query)
        query_document = model.encode(query)
    
        query_document = query_document.reshape(1, -1)
        
        start_time = time.time()
        
        distances, indices = model_knn.kneighbors(query_document, k)

        end_time = time.time()
        times.append(end_time - start_time)

        retrieved_docs =  {anthology_sample[int(index)]['acl_id'] for index in indices.flatten()}
        
        count = 0
        correct = 0
        for i in retrieved_docs:
            if i in relevant_docs:
                correct = 1
                count +=1
        
        accuracy = count / len(retrieved_docs)
        #only evaluate queries that have a solution 
        if relevant_docs :
            distance_values.append(distances.mean())
            accuracy_values.append(accuracy)
            correct_values.append(correct)

            # Precision
            tp = len(relevant_docs & retrieved_docs)
            fp = len(retrieved_docs) - tp
            precision = tp / (tp + fp)
            precision_values.append(precision)
            # Recall
            recall = tp / len(relevant_docs)
            recall_values.append(recall)
        
            # F1 Score
            beta = 1
            f1 = 0
            if recall or precision:
                f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            f1_values.append(f1)

            retrieved_docs =  [anthology_sample[int(index)]['acl_id'] for index in indices.flatten()]

            ap = 0
            count = 0
            if relevant_docs and tp:
                for i in range(len(retrieved_docs)):
                    if retrieved_docs[i] in relevant_docs:
                        count +=1
                        ap += (count/(i+1))
                ap = ap/tp
            ap_values.append(ap)

            with open(query_results_file, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query_dict["q"], 
                                 [[round(val, 4) for val in sublist] for sublist in distances],
                                 round(precision, 4), 
                                 round(recall, 4), 
                                 round(f1, 4)])


    average_distance = sum(distance_values) / len(distance_values)
    average_precision = sum(precision_values) / len(precision_values)
    average_correct = sum(correct_values) / len(correct_values)
    average_recall = sum(recall_values) / len(recall_values)
    average_f1 = sum(f1_values) / len(f1_values)
    average_map = sum(ap_values) / len(ap_values)
    average_time = sum(times) / len(times)
    
    overall_summary_file = os.path.join(output_directory, overall_summary_file)
    with open(overall_summary_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance", "MAP", "Precision", "Recall", "F1_Score", "Correct", "Speed"])
        writer.writerow([round(average_distance, 4), 
                         round(average_map, 4), 
                         round(average_precision, 4),
                         round(average_recall, 4), 
                         round(average_f1, 4), 
                         round(average_correct, 4),
                         round(average_time, 6)])
    print(f'saved results to: {output_directory}')

# Evaluate the performance of the IR system with accelerated nn
def eval_ir_fast(index, k, keys, queries, model, anthology_sample):

    query_results_file = f'query_results_{k}_{keys}' + ".csv"
    
    overall_summary_file = f'overall_summary_{k}_{keys}' + ".csv"
    output_directory = f'./results_2/fast/{k}/{keys}'

    if os.path.exists(output_directory) and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    query_results_file = os.path.join(output_directory, query_results_file)
    with open(query_results_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Distances", "Precision", "Recall", "F1_Score"])

    accuracy_values = []
    distance_values = []
    precision_values = []
    correct_values = []
    recall_values  = []
    f1_values = []
    ap_values = []
    times = []

    for query_dict in queries['queries']:
        query = query_dict['q']
        relevant_docs = set(query_dict['r'])

        query = neural.preprocess(query)
        query_document = model.encode(query)
    
        query_document = query_document.reshape(1, -1)
        
        start_time = time.time()

        indices, distances = index.query(query_document, k=k)
        
        end_time = time.time()
        times.append(end_time - start_time)

        retrieved_docs =  {anthology_sample[int(index)]['acl_id'] for index in indices.flatten()}
        
        count = 0
        correct = 0
        for i in retrieved_docs:
            if i in relevant_docs:
                correct = 1
                count +=1
        
        accuracy = count / len(retrieved_docs)

        if relevant_docs :
            distance_values.append(distances.mean())
            accuracy_values.append(accuracy)
            correct_values.append(correct)

            # Precision
            tp = len(relevant_docs & retrieved_docs)
            fp = len(retrieved_docs) - tp
            precision = tp / (tp + fp)
            precision_values.append(precision)
            # Recall
            recall = tp / len(relevant_docs)
            recall_values.append(recall)
        
            # F1 Score
            beta = 1
            f1 = 0
            if recall or precision:
                f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            f1_values.append(f1)

            # AP
            retrieved_docs =  [anthology_sample[int(index)]['acl_id'] for index in indices.flatten()]

            ap = 0
            count = 0
            if relevant_docs and tp:
                for i in range(len(retrieved_docs)):
                    if retrieved_docs[i] in relevant_docs:
                        count +=1
                        ap += (count/(i+1))
                ap = ap/tp
            ap_values.append(ap)

            with open(query_results_file, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query_dict["q"], 
                                 [[round(val, 4) for val in sublist] for sublist in distances],
                                 round(precision, 4), 
                                 round(recall, 4), 
                                 round(f1, 4)])


    average_distance = sum(distance_values) / len(distance_values)
    average_precision = sum(precision_values) / len(precision_values)
    average_correct = sum(correct_values) / len(correct_values)
    average_recall = sum(recall_values) / len(recall_values)
    average_f1 = sum(f1_values) / len(f1_values)
    average_map = sum(ap_values) / len(ap_values)
    average_time = sum(times) / len(times)
    
    print(f'avg retrieval time, key={keys}, k={k}: {average_time}')
    overall_summary_file = os.path.join(output_directory, overall_summary_file)
    with open(overall_summary_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance", "MAP", "Precision", "Recall", "F1_Score", "Correct", "Speed"])
        writer.writerow([round(average_distance, 4), 
                         round(average_map, 4), 
                         round(average_precision, 4),
                         round(average_recall, 4), 
                         round(average_f1, 4), 
                         round(average_correct, 4),
                         round(average_time, 6)])
    print(f'saved results to: {output_directory}')
        
# Evaluate the performance of the IR system with summarized documents
def eval_ir_sum(model_knn, k, keys, queries, model, anthology_sample):

    query_results_file = f'query_results_{k}_{keys}' + ".csv"
    
    overall_summary_file = f'overall_summary_{k}_{keys}' + ".csv"
    output_directory = f'./results_2/sum/{k}/{keys}'

    if os.path.exists(output_directory) and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    query_results_file = os.path.join(output_directory, query_results_file)
    with open(query_results_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Distances", "Precision", "Recall", "F1_Score"])

    accuracy_values = []
    distance_values = []
    precision_values = []
    correct_values = []
    recall_values  = []
    f1_values = []
    ap_values = []
    times = []
    for query_dict in queries['queries']:
        query = query_dict['q']
        relevant_docs = set(query_dict['r'])

        query = neural.preprocess(query)
        query_document = model.encode(query)
    
        query_document = query_document.reshape(1, -1)
        
        start_time = time.time()
        
        distances, indices = model_knn.kneighbors(query_document, k)

        end_time = time.time()
        times.append(end_time - start_time)

        retrieved_docs =  {anthology_sample[int(index)]['acl_id'] for index in indices.flatten()}
        
        count = 0
        correct = 0
        for i in retrieved_docs:
            if i in relevant_docs:
                correct = 1
                count +=1
        
        accuracy = count / len(retrieved_docs)
        #only evaluate queries that have a solution 
        if relevant_docs :
            distance_values.append(distances.mean())
            accuracy_values.append(accuracy)
            correct_values.append(correct)

            # Precision
            tp = len(relevant_docs & retrieved_docs)
            fp = len(retrieved_docs) - tp
            precision = tp / (tp + fp)
            precision_values.append(precision)
            # Recall
            recall = tp / len(relevant_docs)
            recall_values.append(recall)
        
            # F1 Score
            beta = 1
            f1 = 0
            if recall or precision:
                f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            f1_values.append(f1)

            # AP
            retrieved_docs =  [anthology_sample[int(index)]['acl_id'] for index in indices.flatten()]

            ap = 0
            count = 0
            if relevant_docs and tp:
                for i in range(len(retrieved_docs)):
                    if retrieved_docs[i] in relevant_docs:
                        count +=1
                        ap += (count/(i+1))
                ap = ap/tp
            ap_values.append(ap)

            with open(query_results_file, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query_dict["q"], 
                                 [[round(val, 4) for val in sublist] for sublist in distances],
                                 round(precision, 4), 
                                 round(recall, 4), 
                                 round(f1, 4)])


    average_distance = sum(distance_values) / len(distance_values)
    average_precision = sum(precision_values) / len(precision_values)
    average_correct = sum(correct_values) / len(correct_values)
    average_recall = sum(recall_values) / len(recall_values)
    average_f1 = sum(f1_values) / len(f1_values)
    average_map = sum(ap_values) / len(ap_values)
    average_time = sum(times) / len(times)
    
    overall_summary_file = os.path.join(output_directory, overall_summary_file)
    with open(overall_summary_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance", "MAP", "Precision", "Recall", "F1_Score", "Correct", "Speed"])
        writer.writerow([round(average_distance, 4), 
                         round(average_map, 4), 
                         round(average_precision, 4),
                         round(average_recall, 4), 
                         round(average_f1, 4), 
                         round(average_correct, 4),
                         round(average_time, 6)])
    print(f'saved results to: {output_directory}')


def eval_ir_neural_macro(model_knn, k, keys, queries, model, anthology_sample):

    query_results_file = f'query_results_{k}_{keys}' + ".csv"
    
    overall_summary_file = f'overall_summary_{k}_{keys}' + ".csv"
    output_directory = f'./results/{k}/{keys}'

    if os.path.exists(output_directory) and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    query_results_file = os.path.join(output_directory, query_results_file)
    with open(query_results_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Distances", "Precision", "Recall", "F1_Score"])

    accuracy_values = []
    distance_values = []
    precision_values = []
    correct_values = []
    recall_values  = []
    f1_values = []
    ap_values = []
    times = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_distance = 0
    correct_values = []
    ap_values = []

    num_queries = len(queries['queries'])
    
    for query_dict in queries['queries']:
        query = query_dict['q']
        relevant_docs = set(query_dict['r'])

        query = preprocess(query)
        query_document = model.encode(query)
    
        query_document = query_document.reshape(1, -1)
        start_time = time.time()
        
        distances, indices = model_knn.kneighbors(query_document, k)

        end_time = time.time()
        times.append(end_time - start_time)        

        retrieved_docs =  {anthology_sample[int(index)]['acl_id'] for index in indices.flatten()}
        
        count = 0
        correct = 0
        for i in retrieved_docs:
            if i in relevant_docs:
                correct = 1
                count +=1
        
        accuracy = count / len(retrieved_docs)
        #only evaluate queries that have a solution 
        if relevant_docs :
            distance_values.append(distances.mean())
            accuracy_values.append(accuracy)
            correct_values.append(correct)

            # Precision
            tp = len(relevant_docs & retrieved_docs)
            fp = len(retrieved_docs) - tp
            fn = len(relevant_docs) - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn

            correct = 1 if any(doc in relevant_docs for doc in retrieved_docs) else 0
            correct_values.append(correct)
            total_distance += distances.mean()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(relevant_docs) if len(relevant_docs) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            retrieved_docs =  [anthology_sample[int(index)]['acl_id'] for index in indices.flatten()]
            
            ap = 0
            count = 0

            if relevant_docs and tp:
                for i in range(len(retrieved_docs)):
                    if retrieved_docs[i] in relevant_docs:
                        count += 1
                        ap += count / (i + 1)
                ap /= tp
            ap_values.append(ap)

            with open(query_results_file, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query_dict["q"],
                                 [[round(val, 4) for val in sublist] for sublist in distances],
                                 round(precision, 4),
                                 round(recall, 4),
                                 round(f1, 4)])

    micro_average_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_average_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_average_f1 = (2 * micro_average_precision * micro_average_recall) / (micro_average_precision + micro_average_recall) if (micro_average_precision + micro_average_recall) > 0 else 0

    average_distance = total_distance / num_queries
    average_correct = sum(correct_values) / num_queries
    average_map = sum(ap_values) / len(ap_values)

    average_time = sum(times) / len(times)
    
    overall_summary_file = os.path.join(output_directory, overall_summary_file)
    with open(overall_summary_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance", "MAP", "Precision", "Recall", "F1_Score", "Correct", "Speed"])
        writer.writerow([round(average_distance, 4),
                         round(average_map, 4),
                         round(micro_average_precision, 4),
                         round(micro_average_recall, 4),
                         round(micro_average_f1, 4),
                         round(average_correct, 4),
                         round(average_time, 6)])


def eval_ir_macro(vector, model_knn, keys, queries, document_ids, k, ng, svd, n_components):

    query_results_file = f'query_results_{k}_{keys}' + ".csv"
    
    overall_summary_file = f'overall_summary_{k}_{keys}' + ".csv"
    if ng:
        output_directory = f'./results/ngram/{k}/{keys}'
    elif svd:
        output_directory = f'./results/svd/svd_{n_components}/{k}/{keys}'
    else:
        output_directory = f'./results/{k}/{keys}'

    if os.path.exists(output_directory) and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    query_results_file = os.path.join(output_directory, query_results_file)
    with open(query_results_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Distances", "Precision", "Recall", "F1_Score"])
    accuracy_values = []
    distance_values = []
    precision_values = []
    correct_values = []
    recall_values  = []
    f1_values = []
    ap_values = []
    for query_dict in queries['queries']:
        query = query_dict['q']
        relevant_docs = set(query_dict['r'])
        if k > 0:
            query = preprocess(query)

            query_vector = vector.transform([" ".join(query)])
            if svd:
                query_vector = svd.transform(query_vector)
            distances, indices = model_knn.kneighbors(query_vector, n_neighbors=k)

            retrieved_docs = {document_ids[i] for i in indices.flatten()}
            count = 0
            correct = 0
            for i in retrieved_docs:
                if i in relevant_docs:
                    correct = 1
                    count +=1
            
            accuracy = count / len(retrieved_docs)
            if relevant_docs :
                distance_values.append(distances.mean())
                accuracy_values.append(accuracy)
                correct_values.append(correct)

                # Precision
                tp = len(relevant_docs & retrieved_docs)
                fp = len(retrieved_docs) - tp
                precision = tp / (tp + fp)
                precision_values.append(precision)
                # Recall
                recall = tp / len(relevant_docs)
                recall_values.append(recall)
            
                # F1 Score
                beta = 1
                f1 = 0
                if recall or precision:
                    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                f1_values.append(f1)
 
                # AP
                retrieved_docs = [document_ids[i] for i in indices.flatten()]
                ap = 0
                count = 0
                if relevant_docs and tp:
                    for i in range(len(retrieved_docs)):
                        if retrieved_docs[i] in relevant_docs:
                            count +=1
                            ap += (count/(i+1))
                    ap = ap/tp
                ap_values.append(ap)

                with open(query_results_file, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([query_dict["q"], 
                                     [[round(val, 4) for val in sublist] for sublist in distances],
                                     round(precision, 4), 
                                     round(recall, 4), 
                                     round(f1, 4)])
                    

    average_distance = sum(distance_values) / len(distance_values)
    average_precision = sum(precision_values) / len(precision_values)
    average_correct = sum(correct_values) / len(correct_values)
    average_recall = sum(recall_values) / len(recall_values)
    average_f1 = sum(f1_values) / len(f1_values)
    average_map = sum(ap_values) / len(ap_values)
    
    overall_summary_file = os.path.join(output_directory, overall_summary_file)
    with open(overall_summary_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance", "MAP", "Precision", "Recall", "F1_Score", "Correct"])
        writer.writerow([round(average_distance, 4), 
                         round(average_map, 4), 
                         round(average_precision, 4),
                         round(average_recall, 4), 
                         round(average_f1, 4), 
                         round(average_correct, 4)])
    


