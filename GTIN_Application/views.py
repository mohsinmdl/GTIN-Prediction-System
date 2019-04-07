# importing required packages
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from . import time_Series_Fb_Prophet

import os,numpy,scipy,pickle,math
import json
from dateutil import relativedelta


def load_data2(path):
    if len(path) == 0:
        print("Path not provided")
        return load_data2()
    else:
        clf = pickle.load(open(path,'rb'))
        return clf






# disabling csrf (cross site request forgery)
@csrf_exempt
def index(request):
    # if post request came
    if request.method == 'POST':
        # getting values from post

        print('Check it..plz')
        test = request.POST.getlist('submit_go_Slected')[0]
        print(test)

        if request.POST.getlist('submit_go_Slected')[0] == 'submitGo':

            file_path = request.POST.getlist('tasks[]')
            aggregate = request.POST.getlist('class_selected')

            file_path=str(file_path[0])
            aggregate=str(aggregate[0])

            if aggregate == 'Weekly':
                aggregate = 'w'

            elif aggregate == 'Daily':
                aggregate = 'd'

            elif aggregate == 'Monthly':
                aggregate = 'm'


            print(aggregate)
            print("output location" + file_path)
            last_Value = time_Series_Fb_Prophet.last_Date(file_path, aggregate)

            print(file_path)
            print(aggregate)

            print('Last Value')
            print(last_Value)



            # # getting predictions from the classifier
            # predicted_data = float("%.2f" %clf.predict(satinput))
            data = {}

            # predicted_data = clf.predict(inputs2)
            print('Predicted Output')
            # print(predicted_data)
            dump_data = numpy.array('hello word').tolist()
            data['result'] =last_Value
            print(dump_data)


            # getting our showdata template
            template = loader.get_template('index.html')

            # returing the template

            return HttpResponse(json.dumps(data),content_type="application/json")

        elif request.POST.getlist('submit_go_Slected')[0] == 'submit':
            print('Lalalaaaaaaaaaaaaaaaaa')

            file_path = request.POST.getlist('tasks[]')
            aggregate = request.POST.getlist('class_selected')
            end_date = request.POST.getlist('end_date')
            last_date = request.POST.getlist('last_date')

            print(end_date)
            print(last_date)

            file_path=str(file_path[0])
            aggregate=str(aggregate[0])

            if aggregate == 'Weekly':
                import datetime

                aggregate = 'w'
                # time_Series_Fb_Prophet.main_method(file_path, aggregate)
                last_Value = last_date

                last_date_split = str(last_date[0]).split('-')
                end_date_split = str(end_date[0]).split('-')


                start_date = datetime.date(int(last_date_split[0]), int(last_date_split[1]), int(last_date_split[2]))
                start_date_monday = (start_date - datetime.timedelta(days=start_date.weekday()))
                end_date = datetime.date(int(end_date_split[0]), int(end_date_split[1]), int(end_date_split[2]))
                num_of_weeks = math.floor((end_date - start_date_monday).days / 7.0)-1

                print('Week :')
                print(num_of_weeks)

            elif aggregate == 'Daily':
                from datetime import datetime

                aggregate = 'd'
                # time_Series_Fb_Prophet.main_method(file_path, aggregate)
                last_Value = last_date

                date_format = "%Y-%m-%d"
                a = datetime.strptime(last_date[0], date_format)
                b = datetime.strptime(end_date[0], date_format)
                delta = b - a
                print('Days : ')
                num_of_weeks = delta.days  # that's it
                print(num_of_weeks)

            elif aggregate == 'Monthly':
                from datetime import datetime

                aggregate = 'm'
                # time_Series_Fb_Prophet.main_method(file_path, aggregate)
                last_Value = last_date
                print('Usman is checking')
                print(last_date[0])
                print(end_date[0])
                print('Usman is checking End')

                date_format = "%Y-%m-%d"
                a = datetime.strptime(last_date[0], date_format)
                b = datetime.strptime(end_date[0], date_format)
                r = relativedelta.relativedelta(b, a)
                print('a')
                print(a)
                print(b)
                print('print r' )
                print(r)

                print('Months : ')
                num_of_weeks = (b.year - a.year) * 12 + b.month - a.month
                print(num_of_weeks)








            time_Series_Fb_Prophet.main_method(file_path, aggregate, num_of_weeks)

            # # getting predictions from the classifier
            # predicted_data = float("%.2f" %clf.predict(satinput))
            data = {}

            # predicted_data = clf.predict(inputs2)

            # print(predicted_data)
            dump_data = numpy.array('hello word').tolist()
            data['result'] =last_Value



            # getting our showdata template
            template = loader.get_template('index.html')

            # returing the template

            return HttpResponse(json.dumps(data),content_type="application/json")



        elif request.POST.getlist('submit_go_Slected')[0] == 'all_gtin':

            file_path = request.POST.getlist('tasks[]')
            aggregate = request.POST.getlist('class_selected')


            file_path = str(file_path[0])
            aggregate = str(aggregate[0])

            if aggregate == 'Weekly':
                aggregate = 'w'

            elif aggregate == 'Daily':
                aggregate = 'd'

            elif aggregate == 'Monthly':
                aggregate = 'm'

            print(file_path)
            print(aggregate)

            last_Value = time_Series_Fb_Prophet.last_Date(file_path, aggregate)


            print('Last Date is : ')
            print(last_Value)

            # # getting predictions from the classifier
            # predicted_data = float("%.2f" %clf.predict(satinput))
            data = {}

            # predicted_data = clf.predict(inputs2)
            # print('Predicted Output')
            # print(predicted_data)
            # dump_data = numpy.array('hello word').tolist()
            data['result'] = last_Value
            # print(dump_data)

            # getting our showdata template
            template = loader.get_template('index.html')

            # returing the template

            return HttpResponse(json.dumps(data), content_type="application/json")

        elif request.POST.getlist('submit_go_Slected')[0] == 'submitPredictions':
            print('Lalalaaaaaaaaaaaaaaaaa')

            file_path = request.POST.getlist('tasks[]')
            aggregate = request.POST.getlist('class_selected')
            duration_steps1 = request.POST.getlist('end_date')
            fixed_index = request.POST.getlist('index')




            file_path=str(file_path[0])
            aggregate=str(aggregate[0])
            duration_steps=int(duration_steps1[0])
            fixed_index2 = int(fixed_index[0])

            print('File Path: ')
            print(file_path)
            print('Aggergate: ')
            print(aggregate)
            print('Duration Steps: ')
            print(duration_steps)
            print('Fixed index is : ')
            print(fixed_index2)

            if fixed_index2 == 0:
                if not os.path.isfile('GTIN_Application/static/csv3/FullFuturePredictions.csv'):
                    print('Hello World')
                else:
                    os.remove('GTIN_Application/static/csv3/FullFuturePredictions.csv')



            if aggregate == 'Weekly':
                aggregate = 'w'
            elif aggregate == 'Daily':
                aggregate = 'd'
            elif aggregate == 'Monthly':
                aggregate = 'm'

            time_Series_Fb_Prophet.all_GTIN_Predictions(file_path, aggregate, duration_steps, fixed_index2)

            # # getting predictions from the classifier
            # predicted_data = float("%.2f" %clf.predict(satinput))
            data = {}

            # predicted_data = clf.predict(inputs2)

            # print(predicted_data)
            dump_data = numpy.array('hello word').tolist()
            data['result'] =dump_data



            # getting our showdata template
            template = loader.get_template('index.html')

            # returing the template

            return HttpResponse(json.dumps(data),content_type="application/json")

    else:
        # if post request is not true
        # returing the form template
        template = loader.get_template('index.html')
        return HttpResponse(template.render())



# disabling csrf (cross site request forgery)
@csrf_exempt
def charts(request):
    # if post request came
    if request.method == 'POST':
        # getting our showdata template
        template = loader.get_template('charts.html')

        # returing the template
        return HttpResponse(template.render())
    else:
        # if post request is not true
        # returing the form template
        template = loader.get_template('charts.html')
        return HttpResponse(template.render())