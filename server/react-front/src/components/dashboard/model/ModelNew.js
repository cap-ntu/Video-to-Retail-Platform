import React from "react";
import Card from "@material-ui/core/Card";
import CardHeader from "@material-ui/core/CardHeader";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/CloseRounded";
import CardContent from "@material-ui/core/CardContent";
import withStyles from "@material-ui/core/styles/withStyles";
import Link from "react-router-dom/es/Link";
import TextField from "@material-ui/core/es/TextField/TextField";
import * as PropTypes from "prop-types";
import MenuItem from "@material-ui/core/MenuItem";
import {FilePond} from "react-filepond";
import Typography from "@material-ui/core/Typography";
import Button from "../../common/Button";
import Form from "../../common/Form";
import RequiredField from "../../common/RequiredField";
import Route from "react-router/Route";

const styles = theme => ({
    root: {
        maxWidth: 720,
        margin: 'auto',
    },
    textField: {
        marginLeft: theme.spacing.unit,
        marginRight: theme.spacing.unit,
        width: 200,
    },
    menu: {
        width: 200,
    },
});

const modelType = ["Object", "Face", "Text", "Scene"];

class ModelNew extends Form {
    state = {
        name: "",
        type: "",
        model: null,
        label: null,
    };

    child = {
        name: React.createRef(),
        type: React.createRef(),
    };

    handleDoneWrapper = history => this.handleSubmitWrapper(() => {
        this.props.handleCreate({...this.state}, () => history.push('./'))
    });

    handleChange = name => (event, callback) => {
        this.setState({[name]: event.target.value}, callback);
    };

    render() {

        const {classes} = this.props;
        const {name, type} = this.state;

        return (
            <Card className={classes.root}>
                <CardHeader action={
                    <IconButton component={props => <Link {...props}/>} to={'./'}>
                        <CloseIcon/>
                    </IconButton>}
                            title="Create a new model"
                />
                <CardContent>
                    <RequiredField initValue={""} ref={this.child.name}>
                        <TextField
                            className={classes.textField}
                            id="name"
                            label="Model name"
                            value={name}
                            onChange={this.handleChange("name")}
                            margin="normal"/>
                    </RequiredField>
                    <RequiredField initValue={""} ref={this.child.type}>
                        <TextField
                            id="type"
                            label="Model type"
                            className={classes.textField}
                            value={type}
                            onChange={this.handleChange('type')}
                            SelectProps={{
                                MenuProps: {
                                    className: classes.menu,
                                },
                            }}
                            helperText="Please select your model type"
                            margin="normal"
                            select
                        >
                            {modelType.map(option =>
                                <MenuItem key={option.toUpperCase()} value={option.toUpperCase()}>
                                    {option}
                                </MenuItem>)
                            }
                        </TextField>
                    </RequiredField>
                    <Typography variant="subtitle1" gutterBottom>
                        Model file
                    </Typography>
                    <FilePond maxFiles={1}
                              onupdatefiles={(items) => {
                                  this.setState({
                                      model: items.map(item => item.file)
                                  });
                              }}
                    />
                    <Typography variant="subtitle1" gutterBottom>
                        Label file
                    </Typography>
                    <FilePond maxFiles={1}
                              onupdatefiles={(items) => {
                                  this.setState({
                                      label: items.map(item => item.file)
                                  });
                              }}
                    />
                    <div style={{display: "flex"}}>
                        <div style={{flexGrow: 1}}/>
                        <Route render={({history}) =>
                            <Button color="primary" onClick={this.handleDoneWrapper(history).bind(this)}>Done</Button>
                        }/>
                    </div>
                </CardContent>
            </Card>
        );
    }
}

ModelNew.propTypes = {
    classes: PropTypes.object.isRequired,
    handleCreate: PropTypes.func.isRequired,
};

export default withStyles(styles)(ModelNew);
